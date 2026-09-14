import pytest
import sys

@pytest.fixture(autouse=True)
def force_deterministic_smac(monkeypatch):
    """
    Running SMAC with n_workers=-1 (all cores) for speed results in undeterministic results.
    Therefore, we force n_workers=1 for the tests
    """
    from gedi.generation import hpo

    original_scenario = hpo.Scenario

    def deterministic_scenario(*args, **kwargs):
        kwargs["n_workers"] = 1
        return original_scenario(*args, **kwargs)

    monkeypatch.setattr(hpo, "Scenario", deterministic_scenario)

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