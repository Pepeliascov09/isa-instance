def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests in a real browser (Playwright; slow)")
