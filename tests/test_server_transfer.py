"""Tests for server_transfer module: config parsing."""

from __future__ import annotations

from pathlib import Path

from linumpy_manual_align.server_transfer import ServerConfig, parse_server_config


class TestParseServerConfig:
    def test_valid_subject_dir(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub-22"
        sub_dir.mkdir()
        config_path = sub_dir / "nextflow.config"
        config_path.write_text("params { output = '.' }")

        cfg = parse_server_config(config_path)
        assert cfg is not None
        assert cfg.subject_id == "sub-22"
        assert cfg.host == "132.207.157.41"
        assert cfg.remote_output == "/scratch/workspace/sub-22/output"

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
