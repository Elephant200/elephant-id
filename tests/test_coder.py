import importlib


def test_coding_package_imports_seek_coder():
    coding = importlib.import_module("elephant_id.coding")

    assert coding.SeekCoder.__name__ == "SeekCoder"
