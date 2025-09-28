def test_optional_import():
    try:
        import ragcore.backends.cpp
    except ImportError:
        pass
    assert True
