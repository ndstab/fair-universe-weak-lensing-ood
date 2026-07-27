from pathlib import Path


def test_required_research_documents_exist() -> None:
    required = [
        Path("README.md"),
        Path("docs/AI_USAGE.md"),
        Path("experiments/README.md"),
    ]
    assert all(path.is_file() for path in required)


def test_readme_contains_verified_entry_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert ".venv/bin/python -m pytest" in readme
    assert ".venv/bin/wl-ood make-dummy" in readme
    assert "PRIVATE" in readme
