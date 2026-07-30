from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
import subprocess


def test_public_data_is_ignored_by_git() -> None:
    probe = Path("public_data") / "git-ignore-probe.npy"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(probe)],
        check=False,
    )
    assert result.returncode == 0


def test_matplotlib_runtime_version_is_pinned() -> None:
    try:
        installed_version = version("matplotlib")
    except PackageNotFoundError:
        installed_version = None

    assert installed_version == "3.11.1"


def test_scipy_runtime_version_is_pinned() -> None:
    assert version("scipy") == "1.18.0"


def test_scipy_is_a_direct_runtime_dependency() -> None:
    project_requirements = requires("fair-universe-wl-ood") or []
    assert "scipy==1.18.0" in project_requirements
