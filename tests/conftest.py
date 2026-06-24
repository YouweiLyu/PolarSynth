"""Pytest fixtures that stub the heavy native dependencies.

Mitsuba / OpenEXR / cv2 are not always available on dev machines; the smoke
tests in this directory want to import the project's pure-Python utilities
without those binaries. We install very minimal stubs so the modules
``import`` cleanly.

If you need to run a real render, run the entry scripts directly in an env
where the binaries are installed — this file is only consulted by pytest.
"""
from __future__ import annotations

import sys
import types


class _Stub(types.ModuleType):
    """Permissive stub: attribute lookups, calls, and items all succeed."""

    def __getattr__(self, name):
        v = _Stub(name)
        setattr(self, name, v)
        return v

    def __call__(self, *_args, **_kwargs):
        return self

    def __getitem__(self, *_args, **_kwargs):
        return self


def _install_native_stubs():
    for name in ('cv2', 'OpenEXR', 'Imath', 'mitsuba'):
        if name not in sys.modules:
            sys.modules[name] = _Stub(name)
    # Mitsuba-specific surface that several of our modules touch at import time.
    mi = sys.modules['mitsuba']

    class _T:
        @staticmethod
        def translate(*_a, **_k): return _T()
        @staticmethod
        def scale(*_a, **_k): return _T()
        @staticmethod
        def rotate(*_a, **_k): return _T()
        @staticmethod
        def look_at(*_a, **_k): return _T()
        matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    mi.ScalarTransform4f = _T
    mi.set_variant = lambda *_a, **_k: None
    mi.load_dict = lambda d: type('S', (), {'aov_names': lambda self: []})()
    mi.Bitmap = object


_install_native_stubs()
