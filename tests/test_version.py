import tomllib
from pathlib import Path

import nanobot


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as handle:
        data = tomllib.load(handle)

    assert nanobot.__version__ == data["project"]["version"]
