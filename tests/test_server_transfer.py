"""Tests for :mod:`linumpy_manual_align.remote` (config parsing and remote script loading)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import linumpy_manual_align
from linumpy_manual_align.remote import (
    ServerConfig,
    _cs_server_script,
    parse_server_config,
)


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


class TestServerConfig:
    def test_dataclass(self) -> None:
        cfg = ServerConfig(host="example.com", remote_output="/data/output", subject_id="sub-01")
        assert cfg.host == "example.com"
        assert cfg.remote_output == "/data/output"
        assert cfg.subject_id == "sub-01"


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
