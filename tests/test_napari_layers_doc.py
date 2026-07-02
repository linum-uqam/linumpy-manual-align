"""EXTR-02: doc/smoke tests for ui/napari_layers NapariLayerLifecycle Protocol."""

from __future__ import annotations

import typing

from linumpy_manual_align.ui import napari_layers as napari_layers_module
from linumpy_manual_align.ui.napari_layers import NapariLayerLifecycle

DOCUMENTED_OPERATIONS = (
    "create_pair_layers",
    "update_pair_data",
    "remove_all_layers",
    "set_overlay_mode",
    "refresh_composite",
)


def test_napari_layer_lifecycle_is_protocol() -> None:
    assert issubclass(NapariLayerLifecycle, typing.Protocol)


def test_documented_operations_present() -> None:
    for name in DOCUMENTED_OPERATIONS:
        assert hasattr(NapariLayerLifecycle, name)


def test_module_docstring_documents_ownership_and_exclusions() -> None:
    doc = napari_layers_module.__doc__ or ""
    assert "PairLoadingMixin" in doc
    assert "OverlayStateMixin" in doc
    assert "CrossSectionMixin" in doc
    assert "pop(0)" in doc or "teardown" in doc.lower()
