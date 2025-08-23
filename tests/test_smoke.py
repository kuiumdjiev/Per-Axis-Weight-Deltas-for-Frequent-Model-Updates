def test_import():
    import bitdelta_pipeline
    assert hasattr(bitdelta_pipeline, "__version__")
