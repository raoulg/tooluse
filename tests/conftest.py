"""Configure pytest"""

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Override pytest's caplog fixture to work with loguru."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to False since we're not using multiprocessing in tests
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True)
def propagate_logs():
    """Fixture to handle --log-cli-level flag with loguru."""
    import logging

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            if logging.getLogger(record.name).isEnabledFor(record.levelno):
                logging.getLogger(record.name).handle(record)

    logger.remove()
    logger.add(PropagateHandler(), format="{message}")
    yield
