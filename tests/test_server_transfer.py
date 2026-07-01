"""Tests for :mod:`linumpy_manual_align.remote` (config parsing and remote script loading)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from unittest.mock import MagicMock

import linumpy_manual_align
import pytest
from linumpy_manual_align.remote import (
    ServerConfig,
    _cs_server_script,
    download_manual_align_package,
    parse_server_config,
    upload_manual_transforms,
)
from linumpy_manual_align.remote import scp_ops


class TestParseServerConfig:
    def test_valid_subject_dir(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub-22"
        sub_dir.mkdir()
        config_path = sub_dir / "nextflow.config"
        config_path.write_text("params { output = '.' }")

        cfg = parse_server_config(config_path)
        assert cfg is not None
        assert cfg.subject_id == "sub-22"
        assert cfg.host == ""
        assert cfg.remote_output == "/scratch/workspace/sub-22/output"
        assert cfg.config_path == config_path

    def test_custom_host(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub-22"
        sub_dir.mkdir()
        config_path = sub_dir / "nextflow.config"
        config_path.write_text("")

        cfg = parse_server_config(config_path, host="10.0.0.1")
        assert cfg is not None
        assert cfg.host == "10.0.0.1"

    def test_different_subject(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub-05"
        sub_dir.mkdir()
        config_path = sub_dir / "nextflow.config"
        config_path.write_text("")

        cfg = parse_server_config(config_path)
        assert cfg is not None
        assert cfg.subject_id == "sub-05"
        assert cfg.remote_output == "/scratch/workspace/sub-05/output"

    def test_nonexistent_config(self, tmp_path: Path) -> None:
        result = parse_server_config(tmp_path / "missing" / "nextflow.config")
        assert result is None

    def test_non_subject_parent(self, tmp_path: Path) -> None:
        """Non-standard parent dir name still works, just with a warning."""
        config_path = tmp_path / "nextflow.config"
        config_path.write_text("")

        cfg = parse_server_config(config_path)
        assert cfg is not None
        # subject_id is the parent dir name regardless
        assert cfg.subject_id == tmp_path.name

    def test_custom_remote_base(self, tmp_path: Path) -> None:
        """remote_base overrides the default /scratch prefix in remote_output."""
        sub_dir = tmp_path / "sub-22"
        sub_dir.mkdir()
        config_path = sub_dir / "nextflow.config"
        config_path.write_text("")

        cfg = parse_server_config(config_path, remote_base="/scratch_nvme")
        assert cfg is not None
        assert cfg.remote_output == "/scratch_nvme/workspace/sub-22/output"

    def test_default_remote_base_is_scratch(self, tmp_path: Path) -> None:
        """Default remote_base is /scratch (backward-compatible)."""
        sub_dir = tmp_path / "sub-10"
        sub_dir.mkdir()
        config_path = sub_dir / "nextflow.config"
        config_path.write_text("")

        cfg = parse_server_config(config_path)
        assert cfg is not None
        assert cfg.remote_output.startswith("/scratch/")


class TestServerConfig:
    def test_dataclass(self) -> None:
        cfg = ServerConfig(host="example.com", remote_output="/data/output", subject_id="sub-01")
        assert cfg.host == "example.com"
        assert cfg.remote_output == "/data/output"
        assert cfg.subject_id == "sub-01"


def _server_config() -> ServerConfig:
    return ServerConfig(
        host="h",
        remote_output="/scratch/workspace/sub-22/output",
        subject_id="sub-22",
    )


def _manual_transforms_tree(tmp_path: Path) -> Path:
    root = tmp_path / "manual_transforms"
    for name in ("slice_z01", "slice_z02", "slice_z99"):
        d = root / name
        d.mkdir(parents=True)
        (d / "transform.tfm").touch()
    return root


class TestUploadManualTransforms:
    @pytest.fixture
    def mocks(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: dict = {"scp_args": None, "scp_called": False}

        def fake_run_scp(args: list[str], description: str, **kwargs: object) -> tuple[bool, str]:
            captured["scp_args"] = args
            captured["scp_called"] = True
            return True, "OK"

        monkeypatch.setattr(scp_ops, "_run_scp", fake_run_scp)
        monkeypatch.setattr(
            scp_ops.subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0)),
        )
        return captured

    def test_explicit_slice_dirs_uploads_only_those(
        self, tmp_path: Path, mocks: dict
    ) -> None:
        root = _manual_transforms_tree(tmp_path)
        cfg = _server_config()

        ok, msg = upload_manual_transforms(
            cfg,
            root,
            slice_dirs=[root / "slice_z01", root / "slice_z02"],
        )

        assert ok is True
        assert "2 transforms" in msg
        scp_args = mocks["scp_args"]
        assert scp_args is not None
        paths = [a for a in scp_args if not a.startswith("-") and ":" not in a]
        assert str(root / "slice_z01") in paths
        assert str(root / "slice_z02") in paths
        assert str(root / "slice_z99") not in paths

    def test_none_slice_dirs_falls_back_to_glob(
        self, tmp_path: Path, mocks: dict
    ) -> None:
        root = _manual_transforms_tree(tmp_path)
        cfg = _server_config()

        ok, msg = upload_manual_transforms(cfg, root, slice_dirs=None)

        assert ok is True
        assert "3 transforms" in msg
        scp_args = mocks["scp_args"]
        assert scp_args is not None
        paths = [a for a in scp_args if not a.startswith("-") and ":" not in a]
        assert str(root / "slice_z01") in paths
        assert str(root / "slice_z02") in paths
        assert str(root / "slice_z99") in paths

    def test_empty_slice_dirs_returns_error_without_scp(
        self, tmp_path: Path, mocks: dict
    ) -> None:
        root = _manual_transforms_tree(tmp_path)
        cfg = _server_config()

        ok, msg = upload_manual_transforms(cfg, root, slice_dirs=[])

        assert ok is False
        assert msg
        assert mocks["scp_called"] is False

    def test_slice_dir_outside_root_is_rejected(
        self, tmp_path: Path, mocks: dict
    ) -> None:
        root = _manual_transforms_tree(tmp_path)
        outside = tmp_path / "elsewhere" / "slice_z05"
        outside.mkdir(parents=True)
        cfg = _server_config()

        ok, msg = upload_manual_transforms(
            cfg,
            root,
            slice_dirs=[root / "slice_z01", outside],
        )

        assert ok is False
        assert str(outside) in msg
        assert mocks["scp_called"] is False


class TestDownloadManualAlignPackage:
    def test_download_archives_on_server_then_extracts_locally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _server_config()
        local_dir = tmp_path / "server_package"
        calls: dict[str, list] = {"ssh": [], "scp": [], "rm_ssh": []}

        def fake_run_ssh(
            server: ServerConfig,
            remote_cmd: str,
            description: str,
            *,
            timeout: int = 120,
        ) -> tuple[bool, str]:
            calls["ssh"].append(remote_cmd)
            if remote_cmd.startswith("rm -f "):
                calls["rm_ssh"].append(remote_cmd)
            return True, "OK"

        def fake_run_scp(
            args: list[str],
            description: str,
            *,
            timeout: int = scp_ops._DEFAULT_SCP_TIMEOUT_SEC,
        ) -> tuple[bool, str]:
            calls["scp"].append(args)
            archive_path = Path(args[1])
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            pkg_root = tmp_path / "build_pkg" / "manual_align_package" / "aips"
            pkg_root.mkdir(parents=True)
            (pkg_root / "aip_001.npz").touch()
            (pkg_root / "aip_002.npz").touch()
            import tarfile

            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(pkg_root.parent, arcname="manual_align_package")
            return True, "OK"

        monkeypatch.setattr(scp_ops, "_run_ssh", fake_run_ssh)
        monkeypatch.setattr(scp_ops, "_run_scp", fake_run_scp)

        ok, msg = download_manual_align_package(cfg, local_dir)

        assert ok is True
        assert "2 AIPs" in msg
        assert calls["ssh"]
        assert "tar -czf" in calls["ssh"][0]
        assert "manual_align_package" in calls["ssh"][0]
        scp_args = calls["scp"][0]
        assert len(scp_args) == 2
        assert scp_args[0].startswith("h:")
        assert scp_args[0].endswith(".tar.gz")
        assert "-r" not in scp_args
        assert (local_dir / "manual_align_package" / "aips").exists()
        assert calls["rm_ssh"]


class TestCsServerScript:
    """``_cs_server_script`` must work from editable installs, wheels, and source trees."""

    def test_loads_via_importlib_resources(self) -> None:
        """The script must be loadable as a package resource (non-editable wheel installs)."""
        script_path = resources.files("linumpy_manual_align").joinpath("remote", "cs_server.py")
        text = script_path.read_text(encoding="utf-8")
        assert len(text) > 100
        assert "read_omezarr" in text
        assert "ready " in text
        assert "for line in sys.stdin" in text

    def test_cs_server_script_matches_package_cs_server_py(self) -> None:
        """``_cs_server_script()`` must match the on-disk ``cs_server.py`` next to the package."""
        pkg_dir = Path(linumpy_manual_align.__file__).resolve().parent
        direct = (pkg_dir / "remote" / "cs_server.py").read_text(encoding="utf-8")
        assert _cs_server_script() == direct

    def test_cs_server_script_is_stable_marker_content(self) -> None:
        """Sanity check that the loader returns the expected remote protocol, not an empty string."""
        s = _cs_server_script()
        assert s.startswith('"""')
        assert "linumpy.io.zarr" in s
        assert "base64.b64encode" in s
