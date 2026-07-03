"""Status bar text and saved flash."""

from __future__ import annotations

from linumpy_manual_align.contracts.models import SEVERITY_ERROR, SEVERITY_INFO, SEVERITY_WARNING
from linumpy_manual_align.contracts import PairSessionState
from linumpy_manual_align.io.transform_io import get_metric, load_pairwise_metrics
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget

_SEVERITY_RANK = {
    SEVERITY_ERROR: 3,
    SEVERITY_WARNING: 2,
    SEVERITY_INFO: 1,
}
_RESUME_PROMOTION_SOURCE = "resume_promotion"


class StatusMixin:
    """Mixin that updates the dock status label from current alignment state."""

    def _set_promoted_message(
        self: ManualAlignWidget,
        text: str | None,
        *,
        severity: str,
        source: str,
    ) -> None:
        if not hasattr(self, "_promoted_messages"):
            self._promoted_messages: dict[str, tuple[str, str]] = {}
        if text is None:
            self._promoted_messages.pop(source, None)
        else:
            self._promoted_messages[source] = (severity, text)
        self._refresh_promoted_banner()

    def _refresh_promoted_banner(self: ManualAlignWidget) -> None:
        if not hasattr(self, "promoted_banner"):
            return
        messages = getattr(self, "_promoted_messages", {})
        if not messages:
            self.promoted_banner.hide()
            return
        winning_source = max(
            messages,
            key=lambda src: _SEVERITY_RANK.get(messages[src][0], 0),
        )
        _severity, text = messages[winning_source]
        self.promoted_banner_label.setText(text)
        self.promoted_banner.show()
        if hasattr(self, "btn_dismiss_banner"):
            if winning_source == _RESUME_PROMOTION_SOURCE:
                self.btn_dismiss_banner.show()
            else:
                self.btn_dismiss_banner.hide()

    def _dismiss_promoted_banner(self: ManualAlignWidget) -> None:
        self._set_promoted_message(
            None,
            severity=SEVERITY_INFO,
            source=_RESUME_PROMOTION_SOURCE,
        )

    def _update_status(self: ManualAlignWidget) -> None:
        identity_label = getattr(self, "pair_identity_label", None)
        detail_label = getattr(self, "pair_detail_label", self.status_label)

        if not self.pairs:
            empty_msg = (
                "<i>No data loaded. Use the Server section to download or launch with --data_package.</i>"
            )
            if identity_label is not None:
                identity_label.setText(empty_msg)
                detail_label.setText("")
            else:
                self.status_label.setText(empty_msg)
            return

        fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()
        scale = 2**self.level

        mode_label = {"xy": "XY", "xz": "XZ", "yz": "YZ"}.get(self._projection_mode, "XY")
        identity_line = (
            f"<b>Pair {self.current_pair_idx + 1}/{len(self.pairs)}: "
            f"z{fid:02d} → z{mid:02d}  [{mode_label}]</b>"
        )
        detail_lines: list[str] = []
        if self.level == 0:
            detail_lines.append(
                f"<i>Full resolution (level 0)</i> — tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°"
            )
        else:
            detail_lines.append(
                f"Working (level {self.level}): tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°"
            )
            detail_lines.append(
                f"Full res (level 0): tx={state.tx * scale:.1f}  ty={state.ty * scale:.1f}  rot={state.rotation:.2f}°"
            )

        offsets = self._current_offsets.get(mid, (0, 0))
        if offsets != (0, 0):
            detail_lines.append(
                f"Z offsets: fixed={offsets[0]}  moving={offsets[1]}  (Δ={offsets[0] - offsets[1]:+d})"
            )

        # Show automated metrics if available
        if mid in self.existing_transforms:
            metrics_path = self.existing_transforms[mid] / "pairwise_registration_metrics.json"
            metrics = load_pairwise_metrics(metrics_path)
            auto_tx = get_metric(metrics, "translation_x")
            auto_ty = get_metric(metrics, "translation_y")
            auto_rot = get_metric(metrics, "rotation")
            auto_mag = get_metric(metrics, "translation_magnitude")
            auto_conf = get_metric(metrics, "registration_confidence")
            auto_zcorr = get_metric(metrics, "z_correlation")

            auto_parts = []
            if auto_tx is not None:
                auto_parts.append(f"tx={auto_tx:.1f}")
            if auto_ty is not None:
                auto_parts.append(f"ty={auto_ty:.1f}")
            if auto_rot is not None:
                auto_parts.append(f"rot={auto_rot:.2f}°")
            if auto_mag is not None:
                auto_parts.append(f"mag={auto_mag:.0f}px")
            if auto_parts:
                detail_lines.append(f"<i>Automated: {', '.join(auto_parts)}</i>")

            quality_parts = []
            if auto_conf is not None:
                quality_parts.append(f"conf={auto_conf:.3f}")
            if auto_zcorr is not None:
                quality_parts.append(f"zcorr={auto_zcorr:.3f}")
            if quality_parts:
                detail_lines.append(f"<i>Quality: {', '.join(quality_parts)}</i>")

        session_line = self._session_status_line(mid) if hasattr(self, "_session_status_line") else None
        session_states = getattr(self, "_session_states", {})
        is_invalid = session_states.get(mid) == PairSessionState.INVALID
        if is_invalid and session_line:
            if hasattr(self, "_set_promoted_message"):
                self._set_promoted_message(
                    session_line,
                    severity=SEVERITY_ERROR,
                    source="invalid_pair",
                )
            else:
                detail_lines.append(session_line)
        else:
            if hasattr(self, "_set_promoted_message"):
                self._set_promoted_message(None, severity=SEVERITY_ERROR, source="invalid_pair")
            if session_line:
                detail_lines.append(session_line)

        if self._saved_flash_mid == mid:
            detail_lines.append("<b style='color: green;'>✓ SAVED</b>")

        if identity_label is not None:
            identity_label.setText(identity_line)
            detail_label.setText("<br>".join(detail_lines))
        else:
            all_lines = [identity_line, *detail_lines]
            self.status_label.setText("<br>".join(all_lines))

    def _on_saved_flash_timeout(self: ManualAlignWidget) -> None:
        """Clear the ephemeral SAVED indicator after the flash timer fires."""
        self._saved_flash_mid = None
        self._update_status()

    def _flash_saved(self: ManualAlignWidget, mid: int) -> None:
        """Show the SAVED indicator for 3 seconds then clear it."""
        self._saved_flash_mid = mid
        self._saved_flash_timer.start(3000)
        self._update_status()
