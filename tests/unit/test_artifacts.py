"""Tests for shortcut_lens.artifacts: LocalStore path helpers, HFHubStore push/pull wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shortcut_lens.artifacts import HFHubStore, LocalStore, build_dir, cache_dir, run_dir


def test_local_store_path_resolves_under_root(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert store.path("runs", "r0") == tmp_path / "runs" / "r0"


def test_local_store_exists(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert not store.exists("runs", "r0")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "r0").touch()
    assert store.exists("runs", "r0")


def test_local_store_ensure_dir_creates_nested_directories(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    created = store.ensure_dir("runs", "r0", "predictions")
    assert created.is_dir()
    assert created == tmp_path / "runs" / "r0" / "predictions"


def test_build_dir_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    expected = tmp_path / "builds" / "planted_pets" / "abcd1234"
    assert build_dir(store, "planted_pets", "abcd1234") == expected


def test_cache_dir_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert cache_dir(store, "clip", "render5678") == tmp_path / "cache" / "clip" / "render5678"


def test_run_dir_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    run_id = "e2-planted_pets-abcd1234-s0"
    assert run_dir(store, run_id) == tmp_path / "runs" / run_id


def test_hf_hub_store_path_helpers_match_local_store(tmp_path: Path) -> None:
    store = HFHubStore(repo_id="user/repo", root=tmp_path)
    assert store.path("runs", "r0") == tmp_path / "runs" / "r0"
    assert not store.exists("runs", "r0")
    created = store.ensure_dir("runs", "r0")
    assert created.is_dir()
    assert store.exists("runs", "r0")


def test_hf_hub_store_reads_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    store = HFHubStore(repo_id="user/repo")
    assert store._token == "secret-token"


def test_hf_hub_store_push_run_uploads_the_run_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = HFHubStore(repo_id="user/repo", root=tmp_path, token="t")
    run_directory = run_dir(store, "run-1")
    run_directory.mkdir(parents=True)
    (run_directory / "manifest.json").write_text("{}")

    calls: list[dict[str, Any]] = []

    class _FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            calls.append({"init_token": token})

        def upload_folder(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)

    store.push_run("run-1")

    upload_call = calls[1]
    assert upload_call["repo_id"] == "user/repo"
    assert upload_call["repo_type"] == "dataset"
    assert upload_call["folder_path"] == str(run_directory)
    assert upload_call["path_in_repo"] == "runs/run-1"


def test_hf_hub_store_pull_run_downloads_into_local_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = HFHubStore(repo_id="user/repo", root=tmp_path, token="t")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs))

    store.pull_run("run-1")

    assert calls[0]["repo_id"] == "user/repo"
    assert calls[0]["local_dir"] == tmp_path
    assert calls[0]["allow_patterns"] == ["runs/run-1/*"]
