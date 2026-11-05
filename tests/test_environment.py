import importlib
import sys


def test_python_is_3_11() -> None:
    assert sys.version_info[:2] == (3, 11)


def test_kotha_package_imports() -> None:
    importlib.import_module("kotha")
