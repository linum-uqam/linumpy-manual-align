"""Session confidence UI: classification refresh, combo prefixes, resume block."""

from __future__ import annotations

from collections import Counter

from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import QApplication

from linumpy_manual_align.contracts import (
    SEVERITY_ERROR,
    PairSessionState,
    PairUploadStatus,
    UPLOAD_NO_OUTPUT_DIR,
    assess_upload_readiness,
    classify_pair_state,
    combo_prefix,
    format_session_summary,
    format_upload_issue_line,
    manual_output_dir,
    validate_manual_output,
)
from linumpy_manual_align.contracts.upload_readiness import _resolve_session_output_dir
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget

_COMBO_STATE_FOREGROUND: dict[PairSessionState, str] = {
    PairSessionState.INVALID: "#c0392b",
    PairSessionState.UNSAVED: "#d68910",
    PairSessionState.UNCHANGED: "#7f8c8d",
    PairSessionState.UPLOADED: "#2980b9",
    PairSessionState.READY: "#27ae60",
    PairSessionState.SAVED_LOCAL: "#2ecc71",
}

_COMBO_STATE_TOOLTIP: dict[PairSessionState, str] = {
    PairSessionState.INVALID: "Invalid output or upload blocked — fix before save/upload",
    PairSessionState.UNSAVED: "Unsaved changes — save before upload",
    PairSessionState.UNCHANGED: "No manual transform saved yet",
    PairSessionState.UPLOADED: "Uploaded to server this session",
    PairSessionState.READY: "Saved locally and ready to upload",
    PairSessionState.SAVED_LOCAL: "Saved locally (server mode: not yet uploaded)",
}


class SessionMixin:
    """Mixin that projects headless session state into the Session group UI."""

    def _classify_session_pairs(self: ManualAlignWidget) -> dict[int, PairSessionState]:
        server_enabled = self.server_config is not None
        status_by_mid: dict[int, PairUploadStatus] = {}
        if server_enabled:
            report = assess_upload_readiness(self.pairs, self.output_dir, self.saved_pairs)
            status_by_mid = {pair.moving_id: pair.status for pair in report.pairs}

        states: dict[int, PairSessionState] = {}
        for _fid, mid in self.pairs:
            in_unsaved = mid in self.unsaved_changes
            in_saved = mid in self.saved_pairs
            in_uploaded = mid in self.uploaded_pairs

            resolved = _resolve_session_output_dir(self.output_dir, mid)
            saved_missing_output = mid in self.saved_pairs and resolved is None
            if resolved is not None and resolved.is_dir():
                errors = [
                    issue
                    for issue in validate_manual_output(resolved, mid)
                    if issue.severity == SEVERITY_ERROR
                ]
            else:
                errors = []

            upload_invalid = (
                server_enabled and status_by_mid.get(mid) == PairUploadStatus.INVALID
            )
            invalid = bool(errors) or upload_invalid or saved_missing_output
            ready = server_enabled and status_by_mid.get(mid) == PairUploadStatus.READY

            states[mid] = classify_pair_state(
                invalid=invalid,
                in_unsaved=in_unsaved,
                in_saved=in_saved,
                in_uploaded=in_uploaded,
                ready=ready,
                server_enabled=server_enabled,
            )
        return states

    def _refresh_session_state(self: ManualAlignWidget) -> None:
        if not self.pairs:
            self._session_states = {}
            self.session_summary_label.setText("No pairs in session")
            self._update_resume_visibility()
            return

        self._session_states = self._classify_session_pairs()
        counts = Counter(self._session_states.values())
        self.session_summary_label.setText(
            format_session_summary(counts, server_enabled=self.server_config is not None)
        )
        self._rebuild_pair_combo_labels()
        self._update_resume_visibility()
        self._update_status()

    def _session_prefix_for(self: ManualAlignWidget, mid: int) -> str:
        state = getattr(self, "_session_states", {}).get(mid)
        if state is None:
            return ""
        return combo_prefix(state)

    def _rebuild_pair_combo_labels(self: ManualAlignWidget) -> None:
        self.pair_combo.blockSignals(True)
        current = self.pair_combo.currentIndex()
        for i, (fid, mid) in enumerate(self.pairs):
            self.pair_combo.setItemText(i, self._pair_label(fid, mid))
        self._apply_combo_state_styling()
        self.pair_combo.setCurrentIndex(current)
        self.pair_combo.blockSignals(False)

    def _apply_combo_state_styling(self: ManualAlignWidget) -> None:
        server_enabled = self.server_config is not None
        states = getattr(self, "_session_states", {})
        for i, (_fid, mid) in enumerate(self.pairs):
            state = states.get(mid)
            if state is None:
                continue
            if not server_enabled and state in (
                PairSessionState.READY,
                PairSessionState.UPLOADED,
            ):
                continue
            color_hex = _COMBO_STATE_FOREGROUND[state]
            tooltip = _COMBO_STATE_TOOLTIP[state]
            self.pair_combo.setItemData(
                i, QBrush(QColor(color_hex)), Qt.ForegroundRole
            )
            self.pair_combo.setItemData(i, tooltip, Qt.ToolTipRole)

    def _session_status_line(self: ManualAlignWidget, mid: int) -> str | None:
        state = self._session_states.get(mid)
        if state is None:
            return None
        if state == PairSessionState.INVALID:
            out_dir = manual_output_dir(self.output_dir, mid)
            resolved = _resolve_session_output_dir(self.output_dir, mid)
            if resolved is not None and resolved.is_dir():
                errors = [
                    issue
                    for issue in validate_manual_output(resolved, mid)
                    if issue.severity == SEVERITY_ERROR
                ]
            else:
                errors = []
            if errors:
                return self._format_save_error(mid, errors, resolved or out_dir)
            server_enabled = self.server_config is not None
            if server_enabled:
                report = assess_upload_readiness(
                    self.pairs, self.output_dir, self.saved_pairs
                )
                pair = next(
                    (p for p in report.pairs if p.moving_id == mid),
                    None,
                )
                if pair and pair.issues:
                    err = next(
                        (i for i in pair.issues if i.severity == SEVERITY_ERROR),
                        None,
                    )
                    if err is not None:
                        return format_upload_issue_line(mid, err, pair.output_dir)
            elif mid in self.saved_pairs and resolved is None:
                return f"z{mid:02d}: (missing) - {UPLOAD_NO_OUTPUT_DIR}: No output directory on disk"
            return None
        if state == PairSessionState.UNSAVED:
            return "Unsaved changes"
        if state == PairSessionState.SAVED_LOCAL:
            return "Saved locally"
        if state == PairSessionState.READY:
            return "Ready to upload"
        if state == PairSessionState.UPLOADED:
            return "Uploaded this session"
        return None

    def _update_resume_visibility(self: ManualAlignWidget) -> None:
        if self.server_config is None:
            self.resume_block.hide()

    def _show_resume_block(
        self: ManualAlignWidget, config_line: str, guidance_text: str
    ) -> None:
        self._resume_config_line = config_line
        self.resume_config_label.setText(config_line)
        self.resume_guidance_label.setText(guidance_text)
        self.resume_block.show()

    def _copy_config_line(self: ManualAlignWidget) -> None:
        QApplication.clipboard().setText(self._resume_config_line)
        self.viewer.status = "Copied config line to clipboard"

    def _dismiss_resume(self: ManualAlignWidget) -> None:
        self.resume_block.hide()

    def _mark_pair_edited(self: ManualAlignWidget, mid: int) -> None:
        self.unsaved_changes.add(mid)
        self.uploaded_pairs.discard(mid)
        self._refresh_session_state()
