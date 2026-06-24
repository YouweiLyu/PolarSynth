"""Import-only smoke tests.

These tests exist because the project has no other automated coverage and the
refactor in Phase 1-3 touched the public surface of several modules. They run
in environments without mitsuba / OpenEXR thanks to ``conftest.py``.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_utils_imports_cleanly():
    for mod in (
        'utils.paths',
        'utils.logging_util',
        'utils.cli',
        'utils.pipeline',
        'utils.process_utils',
        'utils.polar_util',
    ):
        importlib.import_module(mod)


@pytest.mark.parametrize('mod_name', [
    'utils.DataSampler',
    'utils.DataSamplerCaseStudy',
    'utils.file_util',
    'utils.render_util',
    'utils.MitsubaRenderer',
    'utils.sampler_util',
    'utils.utils',
])
def test_utils_heavy_modules_import(mod_name):
    importlib.import_module(mod_name)


def _public_class_methods(path):
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                m.name for m in node.body
                if isinstance(m, ast.FunctionDef) and not m.name.startswith('_')
            }
    return out


def test_datasampler_class_surface_unchanged():
    """DataSampler should still expose its top-level classes; method counts
    are guarded so accidental deletions in refactors are caught.
    """
    classes = _public_class_methods(REPO / 'utils' / 'DataSampler.py')
    assert {'DataSampler', 'ModelSampler', 'SceneSampler', 'CameraInfo', 'ModelInfo'} <= set(classes)
    assert len(classes['DataSampler']) >= 25
    assert len(classes['ModelSampler']) >= 10
    assert len(classes['SceneSampler']) >= 8


def test_mitsuba_renderer_has_both_render_paths():
    """Phase 3 added render_inprocess alongside render_subprocess; render()
    dispatches based on the ``inprocess`` flag passed to __init__.
    """
    cls = _public_class_methods(REPO / 'utils' / 'MitsubaRenderer.py')['MitsubaRenderer']
    assert 'render' in cls
    assert 'render_subprocess' in cls
    assert 'render_inprocess' in cls


def test_pipeline_helpers_exist():
    pipeline = importlib.import_module('utils.pipeline')
    for fn in ('render_views', 'run_blender', 'select_cam_infos', 'numbered_prefix'):
        assert hasattr(pipeline, fn), fn


def test_paths_env_var_resolution(monkeypatch):
    monkeypatch.setenv('MITSUBA_ASSETS_DIR', '/tmp/fake-assets')
    # Re-import to pick up the env var
    import importlib
    import utils.paths as p
    importlib.reload(p)
    assert p.ASSETS_DIR == '/tmp/fake-assets'
    assert p.MATERIALS_DIR == '/tmp/fake-assets/materials'
    assert p.HDRI_DIR == '/tmp/fake-assets/hdri'
