from pathlib import Path

from scripts.dexjoco_async import multi_gpu_eval_utils


def test_locate_conda_sh_from_active_python_when_conda_is_not_on_path(
    monkeypatch, tmp_path: Path
) -> None:
    base = tmp_path / "miniconda3"
    conda_sh = base / "etc" / "profile.d" / "conda.sh"
    conda_sh.parent.mkdir(parents=True)
    conda_sh.touch()
    python = base / "envs" / "residual" / "bin" / "python3.10"
    python.parent.mkdir(parents=True)
    python.touch()

    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(multi_gpu_eval_utils.sys, "executable", str(python))
    monkeypatch.setattr(
        multi_gpu_eval_utils.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert multi_gpu_eval_utils.locate_conda_sh() == conda_sh
