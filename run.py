"""Cross-platform entry point for SpeechLens project tasks."""

import argparse
import os
import subprocess
import sys


TASKS = (
    "setup", "test", "dataset", "eval", "eval_test", "score_eval",
    "precompute_demo", "reproduce", "serve",
)
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


def run_task(
    task: str,
    smoke: bool = False,
    passage_id: str | None = None,
) -> None:
    """Run a supported task or report its placeholder status."""
    if task == "setup":
        run_setup()
    elif task == "test":
        command = [sys.executable, "-m", "pytest"]
        if os.environ.get("CI", "").casefold() == "true":
            command.extend(["-m", "not slow"])
        subprocess.run(command, check=True)
    elif task == "dataset":
        subprocess.run(
            [sys.executable, "-m", "scripts.make_dataset"],
            check=True,
        )
    elif task == "eval":
        command = [sys.executable, "-m", "scripts.eval_detection"]
        if smoke:
            command.append("--smoke")
        subprocess.run(
            command,
            check=True,
        )
    elif task == "eval_test":
        subprocess.run(
            [sys.executable, "-m", "scripts.eval_detection_test"],
            check=True,
        )
    elif task == "score_eval":
        subprocess.run(
            [sys.executable, "-m", "scripts.score_eval"],
            check=True,
        )
    elif task == "precompute_demo":
        command = [sys.executable, "-m", "scripts.precompute_demo"]
        if passage_id:
            command.extend(["--passage-id", passage_id])
        subprocess.run(command, check=True)
    elif task == "serve":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "speechlens.api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            check=True,
        )
    else:
        print("not implemented yet")


def main() -> None:
    """Parse the requested task and execute it."""
    parser = argparse.ArgumentParser(description="SpeechLens project tasks")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--passage-id")
    arguments = parser.parse_args()
    if arguments.smoke and arguments.task != "eval":
        parser.error("--smoke is only valid with the eval task")
    if arguments.passage_id and arguments.task != "precompute_demo":
        parser.error("--passage-id is only valid with the precompute_demo task")
    run_task(arguments.task, smoke=arguments.smoke, passage_id=arguments.passage_id)


if __name__ == "__main__":
    main()