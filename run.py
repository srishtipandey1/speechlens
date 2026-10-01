"""Cross-platform entry point for SpeechLens project tasks."""

import argparse
import subprocess
import sys


TASKS = ("setup", "test", "dataset", "eval", "reproduce", "serve")
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def run_setup() -> None:
    """Install CPU-only PyTorch wheels and the pinned project dependencies."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            CPU_TORCH_INDEX,
            "torch==2.6.0",
            "torchaudio==2.6.0",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
    )


def run_task(task: str) -> None:
    """Run a supported task or report its placeholder status."""
    if task == "setup":
        run_setup()
    elif task == "test":
        subprocess.run([sys.executable, "-m", "pytest"], check=True)
    elif task == "dataset":
        subprocess.run(
            [sys.executable, "-m", "scripts.make_dataset"],
            check=True,
        )
    else:
        print("not implemented yet")


def main() -> None:
    """Parse the requested task and execute it."""
    parser = argparse.ArgumentParser(description="SpeechLens project tasks")
    parser.add_argument("task", choices=TASKS)
    arguments = parser.parse_args()
    run_task(arguments.task)


if __name__ == "__main__":
    main()