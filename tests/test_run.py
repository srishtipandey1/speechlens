"""Task-runner behavior tests."""

import run


def test_ci_test_task_excludes_slow_marker(monkeypatch) -> None:
    """CI gets the fast suite while preserving the standard task runner."""
    commands = []
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    run.run_task("test")

    assert commands[0][0][-3:] == ["pytest", "-m", "not slow"]
    assert commands[0][1] is True


def test_local_test_task_keeps_slow_tests_enabled(monkeypatch) -> None:
    """The normal local test task runs pytest without marker filtering."""
    commands = []
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    run.run_task("test")

    assert commands[0][0] == [run.sys.executable, "-m", "pytest"]
    assert commands[0][1] is True
