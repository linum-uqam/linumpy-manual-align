"""SCP download/upload and server config UI."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QFileDialog

from linumpy_manual_align.io.transform_io import discover_aips, discover_pair_aips, discover_transforms
from linumpy_manual_align.remote import ScpWorker
from linumpy_manual_align.settings import settings
from linumpy_manual_align.settings_runtime import (
    default_host_display,
    persist_dock_host_if_changed,
    sync_server_config_host_from_ui,
)
from linumpy_manual_align.ui.widget_typing import ManualAlignWidget


class ServerMixin:
    def _download_from_server(self: ManualAlignWidget) -> None:
        """Download the manual alignment data package from the server (in background thread)."""
        from linumpy_manual_align.remote import download_manual_align_package

        if self.server_config is None:
            return

        self.btn_download.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.server_progress.show()
        self.server_status_label.setText("<i>Downloading...</i>")
        self.viewer.status = "Downloading data from server..."

        # Always download to the same fixed root so repeated downloads overwrite
        # in-place instead of nesting deeper (aips_dir.parent would keep moving
        # deeper each time because aips_dir is updated after every download).
        local_dir = self.output_dir.parent / "server_package"

        self._worker = ScpWorker(download_manual_align_package, (self.server_config, local_dir, self.level))
        self._worker.finished.connect(lambda ok, msg: self._on_download_finished(ok, msg, local_dir))
        self._worker.start()

    def _on_download_finished(self: ManualAlignWidget, ok: bool, msg: str, local_dir: Path) -> None:
        """Handle completion of background download."""
        self.server_progress.hide()
        self.server_status_label.setText(msg)
        self.viewer.status = msg
        self.btn_download.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self._worker = None

        if ok:
            pkg_aips = local_dir / "manual_align_package" / "aips"
            if not pkg_aips.exists():
                pkg_aips = local_dir / "aips"
            if pkg_aips.exists():
                self.aips_dir = pkg_aips
                self.slice_paths = discover_aips(pkg_aips)
                self.slice_ids = list(self.slice_paths.keys())
                self.pair_paths_xy = discover_pair_aips(pkg_aips)

                # Discover axis-specific AIPs
                self._discover_axis_aip_dirs(pkg_aips.parent)

                # Load remote zarr metadata for interactive cross-section sliders
                self._load_remote_cs_metadata(pkg_aips.parent)

                # Reload transforms too if available
                pkg_tfm = local_dir / "manual_align_package" / "transforms"
                if not pkg_tfm.exists():
                    pkg_tfm = local_dir / "transforms"
                if pkg_tfm.exists():
                    self.transforms_dir = pkg_tfm
                    self.existing_transforms = discover_transforms(pkg_tfm)
                # Rebuild pairs and reload
                self._rebuild_pairs()

    def _upload_to_server(self: ManualAlignWidget) -> None:
        """Upload saved manual transforms to the server (in background thread)."""
        from linumpy_manual_align.remote import upload_manual_transforms

        if self.server_config is None:
            return

        if not self.output_dir.exists() or not list(self.output_dir.glob("slice_z*")):
            self.server_status_label.setText("No saved transforms to upload")
            return

        self.btn_download.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.server_progress.show()
        self.server_status_label.setText("<i>Uploading...</i>")
        self.viewer.status = "Uploading transforms to server..."

        self._worker = ScpWorker(upload_manual_transforms, (self.server_config, self.output_dir))
        self._worker.finished.connect(self._on_upload_finished)
        self._worker.start()

    def _on_upload_finished(self: ManualAlignWidget, ok: bool, msg: str) -> None:
        """Handle completion of background upload."""
        self.server_progress.hide()
        self.server_status_label.setText(msg)
        self.viewer.status = msg
        self.btn_download.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self._worker = None

    def _rebuild_pairs(self: ManualAlignWidget) -> None:
        """Rebuild pair list after slice discovery changes, then reload UI."""
        self._build_pairs()

        if not self.pairs:
            self.viewer.status = "No slice pairs found after download"
            return

        # Rebuild combo box
        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        for fid, mid in self.pairs:
            self.pair_combo.addItem(self._pair_label(fid, mid))
        self.pair_combo.blockSignals(False)

        self.current_pair_idx = 0
        self._load_pair(0)

    def _browse_server_config(self: ManualAlignWidget) -> None:
        """Open file dialog to select a nextflow.config for server access."""
        start_dir = str(Path.home() / "Downloads")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select nextflow.config",
            start_dir,
            "Config files (*.config);;All files (*)",
        )
        if not path:
            return
        self._on_config_selected(Path(path))

    def _on_config_selected(self: ManualAlignWidget, config_path: Path) -> None:
        """Parse a nextflow.config and enable server features."""
        from linumpy_manual_align.remote import parse_server_config

        host = self.host_edit.text().strip() or default_host_display()
        cfg = parse_server_config(config_path, host=host)
        if cfg is None:
            self.server_status_label.setText("<b style='color: red;'>Failed to parse config</b>")
            return

        self.server_config = cfg
        self.config_path_edit.setText(str(config_path))
        self.host_edit.blockSignals(True)
        self.host_edit.setText(cfg.host)
        self.host_edit.blockSignals(False)
        settings.set("server/default_host", cfg.host.strip())
        self.btn_download.setEnabled(True)
        self.btn_upload.setEnabled(True)

        # Re-anchor output_dir to the config's parent so paths resolve correctly
        self.output_dir = config_path.parent / "manual_transforms"

        # Check if a package was already downloaded for this config
        existing = self._find_existing_package()
        if existing is not None:
            self.server_status_label.setText(
                f"Configured: {cfg.subject_id} @ {cfg.host} — existing package found at {existing}, loading…"
            )
            self._load_existing_package(existing)
        else:
            self.server_status_label.setText(f"Configured: {cfg.subject_id} @ {cfg.host}")

    def _find_existing_package(self: ManualAlignWidget) -> Path | None:
        """Return the aips/ dir of an already-downloaded package, or None."""
        candidates = [
            self.output_dir.parent / "server_package" / "manual_align_package" / "aips",
            self.output_dir.parent / "server_package" / "aips",
        ]
        for path in candidates:
            if path.exists() and any(path.glob("*.npz")):
                return path
        return None

    def _load_existing_package(self: ManualAlignWidget, aips_dir: Path) -> None:
        """Load a previously downloaded package without hitting the server."""
        # Update pyramid level from package metadata so status display is correct.
        metadata_path = aips_dir.parent / "manual_align_metadata.json"
        if metadata_path.exists():
            import json

            try:
                meta = json.loads(metadata_path.read_text())
                if "pyramid_level" in meta:
                    self.level = int(meta["pyramid_level"])
            except Exception:
                pass

        self.aips_dir = aips_dir
        self.slice_paths = discover_aips(aips_dir)
        self.slice_ids = list(self.slice_paths.keys())
        self.pair_paths_xy = discover_pair_aips(aips_dir)

        # Discover axis-specific AIPs
        self._discover_axis_aip_dirs(aips_dir.parent)

        # Load transforms alongside the aips if present
        for tfm_candidate in (aips_dir.parent / "transforms", aips_dir.parent.parent / "transforms"):
            if tfm_candidate.exists():
                self.transforms_dir = tfm_candidate
                self.existing_transforms = discover_transforms(tfm_candidate)
                break

        # Load remote zarr metadata for interactive cross-section mode
        self._load_remote_cs_metadata(aips_dir.parent)

        self._refresh_saved_pairs()
        self._rebuild_pairs()

    def _on_host_changed(self: ManualAlignWidget, text: str) -> None:
        """Update server config host when the user edits the host field."""
        if self.server_config is not None:
            self.server_config.host = text.strip()
        self._host_persist_timer.start(450)

    def _persist_server_host(self: ManualAlignWidget) -> None:
        """Write the dock Host field to QSettings so it survives restarts."""
        persist_dock_host_if_changed(self.host_edit)

    def _sync_server_config_host_from_ui(self: ManualAlignWidget) -> None:
        """Align ``ServerConfig.host`` with the dock when CLI parsed an empty host.

        ``parse_server_config`` does not read the hostname from nextflow.config;
        ``--server_config`` startup used to leave ``host`` blank while the dock
        showed QSettings — SSH then targeted an empty hostname.
        """
        sync_server_config_host_from_ui(self.server_config, self.host_edit)
