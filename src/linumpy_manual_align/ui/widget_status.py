"""Status bar text and saved flash."""

from __future__ import annotations

from linumpy_manual_align.io.transform_io import get_metric, load_pairwise_metrics
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class StatusMixin:
    def _update_status(self: ManualAlignWidget) -> None:
        if not self.pairs:
            self.status_label.setText(
                "<i>No data loaded. Use the Server section to download or launch with --data_package.</i>"
            )
            return
        fid, mid = self.pairs[self.current_pair_idx]
        state = self._current_state()
        scale = 2**self.level

        mode_label = {"xy": "XY", "xz": "XZ", "yz": "YZ"}.get(self._projection_mode, "XY")
        lines = [f"<b>Pair {self.current_pair_idx + 1}/{len(self.pairs)}: z{fid:02d} → z{mid:02d}  [{mode_label}]</b>"]
        if self.level == 0:
            lines.append(f"tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°")
        else:
            lines.append(f"Working (level {self.level}): tx={state.tx:.1f}  ty={state.ty:.1f}  rot={state.rotation:.2f}°")
            lines.append(
                f"Full res (level 0): tx={state.tx * scale:.1f}  ty={state.ty * scale:.1f}  rot={state.rotation:.2f}°"
            )

        offsets = self._current_offsets.get(mid, (0, 0))
        if offsets != (0, 0):
            lines.append(f"Z offsets: fixed={offsets[0]}  moving={offsets[1]}  (Δ={offsets[0] - offsets[1]:+d})")

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
                lines.append(f"<i>Automated: {', '.join(auto_parts)}</i>")

            quality_parts = []
            if auto_conf is not None:
                quality_parts.append(f"conf={auto_conf:.3f}")
            if auto_zcorr is not None:
                quality_parts.append(f"zcorr={auto_zcorr:.3f}")
            if quality_parts:
                lines.append(f"<i>Quality: {', '.join(quality_parts)}</i>")

        if self._saved_flash_mid == mid:
            lines.append("<b style='color: green;'>✓ SAVED</b>")

        self.status_label.setText("<br>".join(lines))

    def _on_saved_flash_timeout(self: ManualAlignWidget) -> None:
        """Clear the ephemeral SAVED indicator after the flash timer fires."""
        self._saved_flash_mid = None
        self._update_status()

    def _flash_saved(self: ManualAlignWidget, mid: int) -> None:
        """Show the SAVED indicator for 3 seconds then clear it."""
        self._saved_flash_mid = mid
        self._saved_flash_timer.start(3000)
        self._update_status()
