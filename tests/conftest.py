def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests in a real browser (Playwright; slow)")
    config.addinivalue_line(
        "markers", "slow: runs the engine again on the four examples (about 2 min)")
