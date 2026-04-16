"""Single import for :class:`~linumpy_manual_align.ui.widget.ManualAlignWidget` in mixin annotations.

Mixins use ``self: ManualAlignWidget`` without repeating a ``TYPE_CHECKING`` block in every module.
At runtime the name is bound to :class:`object` (annotations are postponed); static checkers use the
``TYPE_CHECKING`` branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linumpy_manual_align.ui.widget import ManualAlignWidget
else:
    ManualAlignWidget = object  # placeholder for import binding only (see TYPE_CHECKING branch)

__all__ = ["ManualAlignWidget"]
