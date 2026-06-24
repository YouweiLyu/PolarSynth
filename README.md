# PolarSynth — Synthetic Polarization Dataset Pipeline

A toolkit for generating large-scale synthetic polarization
datasets (Stokes-vector EXR + per-view geometry / material AOVs) by combining
[Mitsuba 3](https://www.mitsuba-renderer.org/) (polarised path tracing) with
[Blender](https://www.blender.org/) (geometry preprocessing, material baking,
and AOV passes).

The pipeline samples a randomised but reproducible combination of:

* meshes (from your OBJ collection),
* PBR materials (displacement + albedo + roughness),
* HDRI environment lighting and/or point/spot/projector/directional lights,
* camera poses (Fibonacci-spiral, hand-crafted hemispherical, planar, ...),

renders each scene through Mitsuba's Stokes integrator, and post-processes the
EXR into the per-polariser-angle PNGs and AOLP/DOLP maps consumed by typical
shape-from-polarization datasets.

---

## Features

* **Four ready-made pipelines** that share the same sampler / pipeline / renderer
  infrastructure:
  * `render_sgl_obj.py` — single object, random material/lighting variants.
  * `render_mult_obj.py` — multi-object scenes with an optional ground plane.
  * `render_metal_dielec_obj.py` — hybrid metal + dielectric scenes
    (two-pass rendering for material-segment masks).
  * `render_case_study.py` — deterministic sweep over a small (material,
    albedo, roughness, HDRI) grid for qualitative comparisons.
* **Two Mitsuba execution modes**:
  * subprocess CLI (default; compatible with all variants),
  * in-process `mi.render()` via `--inprocess` (skips per-view process spawn
    + JIT warmup).
* **Blender side**: Cycles AOV passes for albedo + roughness + normal + depth
  + per-material masks (metal / dielectric).
* **Configurable via environment variables only** — no hardcoded user paths.
* **12 smoke tests** runnable without Mitsuba / Blender installed.

---

## Requirements

| Component | Tested version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | |
| [Mitsuba 3](https://www.mitsuba-renderer.org/) | 3.3 – 3.5 | Compiled with the `..._spectral_polarized` variant you intend to use. |
| [Blender](https://www.blender.org/) | 3.6 LTS | Must be on `$PATH` as `blender`. |
| numpy / opencv-python / matplotlib / tqdm | see `requirements.txt` | |
| OpenEXR + Imath | 1.3 / 0.0.3 | For multi-layer EXR I/O. |

GPU acceleration is optional:
* For Mitsuba: install the CUDA wheel and pick `--mi_variant cuda_spectral_polarized`.
* For Blender: Cycles uses GPU automatically when `scene.cycles.device = 'GPU'`
  is honoured (the postprocess scripts set this for you).

---

## Installation

```bash
git clone https://github.com/youweilyu/PolarSynth.git
cd PolarSynth

# Python deps. Install Mitsuba separately if you need a custom build.
pip install -r requirements.txt

# Mitsuba CLI on PATH (only required for the default subprocess render mode)
source /path/to/mitsuba3/build/setpath.sh

# Verify Blender is reachable
blender --version
```

Copy `.env.example` to `.env` and adjust the asset paths to match your machine:

```bash
cp .env.example .env
$EDITOR .env
# then before running any pipeline:
set -a; source .env; set +a
```

---

## Asset layout

The pipeline does **not** ship any 3D assets — those are large and licensed
externally. You provide your own under `MITSUBA_ASSETS_DIR` (defaults to
`./assets/3D_assets`). Expected layout:

```
$MITSUBA_ASSETS_DIR/
├── hdri/
│   └── polyhaven/
│       ├── art_studio_2k.exr
│       └── ...
├── materials/
│   └── ambientcg/
│       └── <MaterialName>/
│           ├── <MaterialName>_4K_Color.jpg
│           ├── <MaterialName>_4K_Roughness.jpg
│           └── <MaterialName>_4K_Displacement.jpg
└── models/
    └── <mesh>.obj
    └── <mesh>_objinfo.npy        # see "objinfo files" below
```

### `*_objinfo.npy` files

Each mesh `<m>.obj` must have a sibling `<m>_objinfo.npy` that is a
`numpy.save`'d Python dict with at minimum:

| Key | Type | Meaning |
|---|---|---|
| `bbox_center` | `np.ndarray (3,)` | World-space centre used for camera framing. |
| `is_symmetric` | `bool` | Reduces sampled view count by 4× when true. |
| `is_flat` | `bool` | Switches to overhead camera sampling. |
| `is_disp_prohibit` | `bool` | Skip displacement augmentation for this mesh. |
| `vertex_num` | `int` | Drives Blender's subdivision level. |

You can generate these from your model collection with
`check_scripts/get_obj_props.py` (edit the search roots at the top of the
file first).

### Path-list `.txt` files

The render scripts take three plain-text lists of paths, one entry per line:

* `--model_txt_path` — paths to `.obj` files.
* `--material_txt_path` — paths to *parent* directories containing
  `<dir>/<MaterialName>/<MaterialName>_4K_Color.jpg` triplets.
* `--hdri_txt_path` — paths to `.exr` HDR images.

Place these under `txts/` (gitignored by default).

---

## Quick start

### 1. Run a smoke render

```bash
python render_sgl_obj.py \
    --model_txt_path        txts/models_demo.txt \
    --material_txt_path     txts/materials_demo.txt \
    --hdri_txt_path         txts/hdri_demo.txt \
    --num_rendered_per_obj  1 \
    --pose_num_per_scene    1 \
    --save_dir              renderings/smoke \
    --cache_dir             tmp/smoke \
    --workers               8 \
    --debug
```

A successful run produces, per camera view, a `<name>_NNN_mi.exr`
multi-layer EXR plus the downstream `_img_mi.jpg` and the four polariser-angle
PNGs (`pol000`, `pol045`, `pol090`, `pol135`) after the EXR conversion pass.

### 2. Render at scale

Driver shell scripts under `run/` show production-scale invocations. Either
edit them to point at your data layout or invoke the Python entry directly,
e.g.:

```bash
python render_mult_obj.py \
    --model_txt_path     txts/models_train.txt \
    --material_txt_path  txts/materials_train.txt \
    --hdri_txt_path      txts/hdri_train.txt \
    --num_obj_per_scene  2 \
    --num_scene          500 \
    --pose_num_per_scene 4 \
    --mesh_superimpose \
    --fov                20 \
    --start_index        1 \
    --save_dir           "$MITSUBA_DATA_DIR/training/multi_obj_2obj" \
    --workers            16
```

### 3. Convert EXR to polariser PNGs

`render_*.py` only produces multi-layer EXR. For downstream training the
typical next step is to invoke one of the converter scripts:

```bash
python check_scripts/convert_renderings.py
```

(Edit `src_dir` / `dst_dir` at the top of the file.)

---

## Pipeline modes

| Entry script | What it samples | Output naming |
|---|---|---|
| `render_sgl_obj.py` | one mesh + optional plane, random material/light per variant | `{scene}_{variant:03d}_{view:03d}` |
| `render_mult_obj.py` | N meshes per scene with optional plane | `{idx:05d}_{N}{names}_{view:03d}` |
| `render_metal_dielec_obj.py` | exactly 2 meshes, one metal + one dielectric, rendered twice (hybrid + dielectric-only) | `{idx:05d}_{N}{names}_hybrid_{view:03d}` and `..._dielec_{view:03d}` |
| `render_case_study.py` | deterministic (material × albedo × roughness × HDRI) sweep over a fixed mesh | `{material}_{albedo}_{roughness}_{envmap}` |

Common flags shared by all four (see `utils/cli.py::add_common_render_args`):

```
--model_txt_path / --material_txt_path / --hdri_txt_path
--save_dir / --cache_dir
--mi_variant {scalar_rgb,scalar_spectral_polarized,llvm_spectral_polarized,cuda_spectral_polarized}
--workers       # threads passed to Mitsuba and Blender
--pose_num_per_scene
--start_model_index
--inprocess     # render via mi.render() instead of the `mitsuba` subprocess
--debug         # DEBUG-level logging + early-stop loops
```

`render_mult_obj.py` and `render_metal_dielec_obj.py` add:
`--num_obj_per_scene`, `--num_scene`, `--start_index`, `--mesh_superimpose`,
`--disable_mesh_disp`, `--fov`.

---

## Environment variables

All resolved through `utils/paths.py`. CLI flags always win where both apply.

| Variable | Default | Purpose |
|---|---|---|
| `MITSUBA_ASSETS_DIR` | `./assets/3D_assets` | Root of input 3D assets (`hdri/`, `materials/`, `models/`). |
| `MITSUBA_HDRI_DIR` | `$MITSUBA_ASSETS_DIR/hdri` | Override just the HDRI subroot. |
| `MITSUBA_MATERIALS_DIR` | `$MITSUBA_ASSETS_DIR/materials` | Override just the materials subroot. |
| `MITSUBA_MODELS_DIR` | `$MITSUBA_ASSETS_DIR/models` | Override just the models subroot. |
| `MITSUBA_DEFAULT_MATERIAL` | `$MITSUBA_MATERIALS_DIR/ambientcg/Candy/Candy001` | Fallback texture set. |
| `MITSUBA_DEFAULT_HDRI` | `$MITSUBA_HDRI_DIR/polyhaven/spiaggia_di_mondello_2k.exr` | Fallback HDRI. |
| `MITSUBA_DATA_DIR` | `./datasets` | Root of *generated* dataset trees (consumed by `check_scripts/` + `run/`). |
| `MITSUBA_RENDER_LOG_LEVEL` | `INFO` | One of DEBUG/INFO/WARNING/ERROR. `--debug` forces DEBUG. |
| `MI_DEFAULT_VARIANT` | `llvm_spectral_polarized` | Default Mitsuba variant for `run/*.sh`. |

---

## Output format

Each rendered view produces:

```
<save_dir>/
├── <scene>_<idx>_mi.exr           # multi-layer EXR: S0/S1/S2 stokes + normal + albedo + depth AOVs
├── <scene>_<idx>_bl.exr           # multi-layer EXR from Blender Cycles (normal/albedo/roughness/mask)
├── <scene>_<idx>_img_mi.jpg       # tone-mapped colour preview
├── <scene>_<idx>_albedo_bl.png    # diffuse colour, sRGB-encoded
├── <scene>_<idx>_rough_bl.png     # roughness, linear-encoded
├── <scene>_<idx>_normal_G_bl.png  # camera-space normals (16-bit)
├── <scene>_<idx>_normal_L_bl.png  # world-space normals (16-bit)
├── <scene>_<idx>_mask_bl.png      # foreground mask
├── <scene>_<idx>_dielec_mask_bl.png  # only multi-obj / hybrid scenes
└── <scene>_<idx>_metal_mask_bl.png   # only multi-obj / hybrid scenes
```

After running `check_scripts/convert_renderings.py` the polariser-angle PNGs
land in `<dst_dir>/pol000/<scene>_<idx>.png` (and `pol045`, `pol090`, `pol135`,
plus `mask/`).

---

## Repository layout

```
.
├── render_sgl_obj.py              # entry: single-object pipeline
├── render_mult_obj.py             # entry: multi-object pipeline
├── render_metal_dielec_obj.py     # entry: hybrid metal+dielec pipeline
├── render_case_study.py           # entry: deterministic sweep
├── blender_postprocess.py         # Blender side for single-object pipeline
├── blender_postprocess_multobj.py # Blender side for multi-object pipelines
├── blender_postprocess_common.py  # shared Blender helpers
├── utils/
│   ├── DataSampler.py             # DataSampler + ModelSampler + SceneSampler
│   ├── DataSamplerCaseStudy.py    # deterministic sampler for case-study pipeline
│   ├── MitsubaRenderer.py         # scene-dict wrapper, subprocess + in-process render paths
│   ├── render_util.py             # BSDF / light dict builders, view-direction samplers
│   ├── file_util.py               # EXR readers, polariser maths, image writers
│   ├── pipeline.py                # shared inner render loop + run_blender helper
│   ├── cli.py                     # shared argparse fragments
│   ├── paths.py                   # env-var-driven asset path resolution
│   ├── logging_util.py            # setup_logging() helper
│   ├── sampler_util.py            # info-dict builders for Blender npy
│   ├── process_utils.py           # subprocess wrapper that exits on non-zero return
│   ├── polar_util.py              # DoP / AoP maths
│   └── utils.py                   # misc helpers (load_material, list_dir, ...)
├── tools/                         # standalone helper scripts (run from repo root)
│   ├── get_objaverse.py           # download Objaverse meshes
│   ├── exr/                       # EXR conversion + AoP/DoP inspection
│   │   ├── exr2img.py             # multi-layer EXR → tonemapped JPG
│   │   ├── exr2polar_img.py       # EXR → per-polariser PNG batch
│   │   ├── exr_converter_blender.py  # Blender-side EXR → PNG (img/albedo/normal)
│   │   ├── test_exr_AoPDoP.py     # plot DoP / AoP for one EXR
│   │   └── test_exr_AoPDoP_channel.py # per-channel DoP / AoP inspector
│   ├── mesh/                      # mesh format conversion
│   │   ├── glb2obj.py             # GLB → OBJ via Blender (with material wiring)
│   │   └── glb2obj_raw.py         # GLB → OBJ Blender addon (raw geometry)
│   └── legacy/                    # historical / one-off scripts kept for reference
│       ├── get_params.py          # Mitsuba 2 enoki inverse-rendering toy
│       ├── render_from_xml.py     # render a hardcoded list of XML scenes
│       └── render_normal_blender_from_mitsuba.py # standalone Blender normal pass
├── scenes/                        # canonical Mitsuba XML scenes
│   └── components/                # reusable sensor / integrator XML fragments
├── xmls/                          # exploratory / one-off Mitsuba XML scenes
├── check_scripts/                 # post-processing batch tools (user-specific)
├── run/                           # example shell drivers
├── tests/                         # 12 smoke + AST tests, runnable without GPU
├── requirements.txt
├── .env.example
└── LICENSE
```

Scripts under `tools/` are designed to be invoked from the repo root, e.g.
`python tools/exr/exr2polar_img.py --data_dir ...`. The ones that import the
`utils/` package add the repo root to `sys.path` automatically.

---

## Performance tips

* **`--inprocess`** avoids spawning a `mitsuba` process per view; biggest
  speedup when `pose_num_per_scene` is large. Requires the matching Mitsuba
  Python wheel.
* **`--mi_variant scalar_rgb`** is the fastest variant — useful for smoke
  tests where polarisation accuracy is not the bottleneck.
* **Cache smoothed meshes**: `blender_postprocess.py` writes
  `<model>_smooth<level>.obj` next to the source mesh and reuses it across
  variants. Avoid wiping the cache between runs of the same mesh.
* **Hoist `MitsubaRenderer`**: already done in this repo (Phase 2 refactor).
  If you fork and add new entry scripts, build the renderer **outside** the
  per-scene loop so the LLVM/CUDA backend warms up only once.

---


## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{lyu2024sfpuel,
    title={{SfPUEL}: Shape from Polarization under Unknown Environment Light},
    author={Youwei Lyu and Heng Guo and Kailong Zhang and Si Li and Boxin Shi},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year={2024},
}
```

…and also cite [Mitsuba 3](https://www.mitsuba-renderer.org/) and
[Blender](https://www.blender.org/).
