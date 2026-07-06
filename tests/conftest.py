import pytest
import sys

@pytest.fixture(scope="session", autouse=True)
def remove_resource_tracker_warnings():
    """
    Prevents macOS/Python spawn multiprocessing from leaking tracking signals
    at the boundary of the test session shutdown by safely draining the tracking cache.
    """

    yield  # Let all tests finish running
    if "multiprocessing.resource_tracker" in sys.modules:
        from multiprocessing import resource_tracker
        tracker = resource_tracker._resource_tracker

        for attr in ["_clearafter", "_cleanups"]:
            if hasattr(tracker, attr):
                getattr(tracker, attr).clear()