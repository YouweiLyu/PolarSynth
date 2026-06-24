"""Environment-driven defaults for asset and dataset paths.

All paths used by the rendering pipeline are resolved through this module so
the codebase has no hardcoded absolute paths. Defaults are relative to the
repository root; override via environment variables to point at your own
asset storage.

Environment variables (with their fallbacks):

    MITSUBA_ASSETS_DIR        -> ./assets/3D_assets
    MITSUBA_HDRI_DIR          -> $MITSUBA_ASSETS_DIR/hdri
    MITSUBA_MATERIALS_DIR     -> $MITSUBA_ASSETS_DIR/materials
    MITSUBA_MODELS_DIR        -> $MITSUBA_ASSETS_DIR/models
    MITSUBA_DEFAULT_MATERIAL  -> $MITSUBA_MATERIALS_DIR/ambientcg/Candy/Candy001
    MITSUBA_DEFAULT_HDRI      -> $MITSUBA_HDRI_DIR/polyhaven/spiaggia_di_mondello_2k.exr
    MITSUBA_DATA_DIR          -> ./datasets

The first three (``ASSETS_DIR``, ``HDRI_DIR``, ``MATERIALS_DIR``) point at the
*input* asset collections (3D models, HDRIs, PBR materials). ``MITSUBA_DATA_DIR``
is consumed by the helper scripts in ``check_scripts/`` and ``run/`` as the
root of generated training/test datasets.

Callers should treat these as defaults only — explicit CLI args still win.
"""
from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


ASSETS_DIR = _env("MITSUBA_ASSETS_DIR", "./assets/3D_assets")
HDRI_DIR = _env("MITSUBA_HDRI_DIR", os.path.join(ASSETS_DIR, "hdri"))
MATERIALS_DIR = _env("MITSUBA_MATERIALS_DIR", os.path.join(ASSETS_DIR, "materials"))
MODELS_DIR = _env("MITSUBA_MODELS_DIR", os.path.join(ASSETS_DIR, "models"))

DEFAULT_MATERIAL_DIR = _env(
    "MITSUBA_DEFAULT_MATERIAL",
    os.path.join(MATERIALS_DIR, "ambientcg", "Candy", "Candy001"),
)
DEFAULT_HDRI_PATH = _env(
    "MITSUBA_DEFAULT_HDRI",
    os.path.join(HDRI_DIR, "polyhaven", "spiaggia_di_mondello_2k.exr"),
)

# Root of *generated* datasets (rendered EXR / PNG output collections), used
# by the post-processing batch scripts under check_scripts/ and run/.
DATA_DIR = _env("MITSUBA_DATA_DIR", "./datasets")
