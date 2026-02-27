"""
pytest plugin: write test results to test_service.log after each test.
"""
import logging

LOG_FILE = "test_service.log"

logger = logging.getLogger("test_service")
_counts = {"passed": 0, "failed": 0, "skipped": 0}


def pytest_configure(config):
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def pytest_runtest_logreport(report):
    if report.when != "call":
        return

    name = report.nodeid
    if report.passed:
        _counts["passed"] += 1
        logger.info("PASSED  %s", name)
    elif report.failed:
        _counts["failed"] += 1
        longrepr = str(report.longrepr) if report.longrepr else ""
        logger.error("FAILED  %s\n%s", name, longrepr)
    elif report.skipped:
        _counts["skipped"] += 1
        logger.warning("SKIPPED %s", name)


def pytest_sessionfinish(session, exitstatus):
    logger.info(
        "Session finished — exit status: %d | passed: %d  failed: %d  skipped: %d",
        int(exitstatus),
        _counts["passed"],
        _counts["failed"],
        _counts["skipped"],
    )
