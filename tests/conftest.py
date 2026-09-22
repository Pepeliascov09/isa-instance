def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: testes de ponta a ponta num navegador real (Playwright; lentos)")
