from __future__ import annotations

from importlib.resources import files


def test_package_imports() -> None:
    import doxa  # noqa: F401


def test_package_declares_inline_types() -> None:
    assert files("doxa").joinpath("py.typed").is_file()
