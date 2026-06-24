"""Shared inner-loop helpers for the rendering pipeline.

Before Phase 2 the same scene-x-view nested loop and the same Blender
subprocess invocation were copy-pasted across render_sgl_obj, render_mult_obj,
render_metal_dielec_obj and render_case_study. They now live here.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Callable, Iterable, Sequence

log = logging.getLogger(__name__)

# Light types that produce one render-per-view per camera.
_ENVMAP_LIKE = ('envmap', 'envmap_otherlights')


def select_cam_infos(light_type: str, scene_idx: int, cam_infos: Sequence[dict]) -> Sequence[dict]:
    """Pick the cameras to use for a given scene/lighting index.

    For envmap (and envmap+other-light) scenes we render every camera per scene
    index. For pure other-lights scenes we render exactly the matching camera.
    """
    if light_type in _ENVMAP_LIKE:
        return cam_infos
    if light_type == 'otherlights':
        return [cam_infos[scene_idx]]
    raise ValueError(f'Invalid light type: {light_type!r}')


def render_views(
    renderer,
    light_info: dict,
    cam_infos: Sequence[dict],
    scene_num: int,
    save_prefix_fn: Callable[[int], str],
) -> int:
    """Run the standard scene-x-view rendering loop.

    Args:
        renderer: a MitsubaRenderer with integrator + model already loaded.
        light_info: dict from SceneSampler / DataSampler ``get_light_info``.
        cam_infos: list of camera-info dicts produced via ``get_cam_info('mi')``.
        scene_num: total scene multiplicity (envmap=1, otherlights=N).
        save_prefix_fn: ``int render_count -> str save_prefix``.

    Returns:
        Number of views actually rendered (used by callers for naming continuity).
    """
    light_type = light_info['light_type']
    render_count = 0
    for scene_idx in range(scene_num):
        if scene_num > 1:
            log.info('  ## Rendering [%4d/%4d] lightings ##', scene_idx + 1, scene_num)
        renderer.load_light(
            light_info['envmap'],
            light_info['otherlights'][scene_idx],
        )
        view_cams = select_cam_infos(light_type, scene_idx, cam_infos)
        for cam_idx, camera_info in enumerate(view_cams):
            if len(view_cams) > 1:
                log.info('  ## Rendering [%4d/%4d] views ##', cam_idx + 1, len(view_cams))
            render_count += 1
            renderer.load_sensor(camera_info)
            renderer.render(save_prefix_fn(render_count))
    return render_count


def run_blender(
    script: str,
    info_path: str,
    save_dir: str,
    workers: int,
    extra_args: Iterable[str] = (),
    check: bool = True,
    quiet_stdout: bool = True,
) -> float:
    """Invoke Blender headless with our standard flag set, return wall time."""
    cmd = [
        'blender', '-b',
        '-t', f'{workers}',
        '--python-exit-code', '1',
        '-P', script,
        '--', info_path,
        '--save_dir', save_dir,
        *extra_args,
    ]
    t0 = time.time()
    subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.DEVNULL if quiet_stdout else None,
    )
    elapsed = time.time() - t0
    log.info('Blender time: %.2fs', elapsed)
    return elapsed


def numbered_prefix(save_dir: str, scene_name: str) -> Callable[[int], str]:
    """Build a callable returning ``{save_dir}/{scene_name}_{nnn:03d}`` prefixes."""
    def _fn(idx: int) -> str:
        return os.path.join(save_dir, f'{scene_name}_{idx:03d}')
    return _fn
