"""Execute a notebook, preserving completed cells even if a later cell fails.

The default mode uses IPython in this process and needs no localhost sockets.
Use --nbclient for an isolated Jupyter kernel with a strict per-cell timeout.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import nbformat


TEXT_PROGRESS_SETUP = """\
import sys as _progress_sys
import types as _progress_types
try:
    import tqdm as _progress_package
    from tqdm import tqdm as _text_tqdm, trange as _text_trange
except ImportError:
    pass
else:
    for _progress_name in ('tqdm.auto', 'tqdm.autonotebook'):
        _progress_module = _progress_types.ModuleType(_progress_name)
        _progress_module.tqdm = _text_tqdm
        _progress_module.trange = _text_trange
        _progress_module.__all__ = ['tqdm', 'trange']
        _progress_sys.modules[_progress_name] = _progress_module
        setattr(_progress_package, _progress_name.rsplit('.', 1)[1], _progress_module)
"""


class CellFailed(RuntimeError):
    """The failing cell's outputs have already been saved in the notebook."""


def save_notebook(notebook, path: Path) -> None:
    """Replace the notebook atomically, so interruptions do not truncate it."""
    pending = path.with_name(path.name + ".executing.tmp")
    try:
        with pending.open("w", encoding="utf-8", newline="\n") as handle:
            nbformat.write(notebook, handle)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def describe(cell) -> str:
    lines = [line.strip() for line in cell.source.splitlines() if line.strip()]
    return (lines[0] if lines else "(empty cell)")[:110]


def report_cell(index: int, total: int, cell) -> float:
    print(f"[{index + 1}/{total}] {cell.cell_type}: {describe(cell)}", flush=True)
    return time.monotonic()


def _append_stream(outputs: list, name: str, value: str) -> None:
    if not value:
        return
    if outputs and outputs[-1].output_type == "stream" and outputs[-1].name == name:
        outputs[-1].text += value
    else:
        outputs.append(nbformat.v4.new_output("stream", name=name, text=value))


class _OutputStream:
    """Capture streams in the same order as rich displays, with a console tee."""

    def __init__(self, outputs: list, name: str, console) -> None:
        self.outputs = outputs
        self.name = name
        self.console = console
        self.encoding = getattr(console, "encoding", "utf-8") or "utf-8"

    def write(self, value: str) -> int:
        _append_stream(self.outputs, self.name, value)
        self.console.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


class _DisplayOutputs(list):
    def __init__(self, outputs: list) -> None:
        super().__init__()
        self.outputs = outputs

    def append(self, item) -> None:
        # capture_output's publisher sends dictionaries containing MIME bundles.
        self.outputs.append(
            nbformat.v4.new_output(
                "display_data", data=item.get("data", {}), metadata=item.get("metadata", {})
            )
        )


def run_cell_in_process(shell, cell, timeout: int) -> None:
    """Execute one cell with IPython magics, rich output, and shared variables."""
    import _thread

    from IPython.utils.capture import capture_output

    outputs = []
    cell.outputs = outputs
    cell.execution_count = shell.execution_count
    console_stdout, console_stderr = sys.stdout, sys.stderr
    hook = shell.displayhook
    old_prompt, old_format = hook.write_output_prompt, hook.write_format_data
    timed_out = threading.Event()

    def capture_result(data, metadata=None):
        outputs.append(
            nbformat.v4.new_output(
                "execute_result",
                execution_count=cell.execution_count,
                data=data,
                metadata=metadata or {},
            )
        )

    def interrupt_cell():
        timed_out.set()
        console_stderr.write(f"\nCell exceeded {timeout}s; interrupting.\n")
        console_stderr.flush()
        _thread.interrupt_main()

    timer = threading.Timer(timeout, interrupt_cell)
    timer.daemon = True
    result = None
    failure = None
    try:
        # Keep the regular display hook's history/result handling, replacing only
        # its terminal rendering. capture_output alone prints Out[n] as stdout.
        hook.write_output_prompt = lambda: None
        hook.write_format_data = capture_result
        with capture_output(stdout=True, stderr=True, display=True):
            shell.display_pub.outputs = _DisplayOutputs(outputs)
            sys.stdout = _OutputStream(outputs, "stdout", console_stdout)
            sys.stderr = _OutputStream(outputs, "stderr", console_stderr)
            timer.start()
            try:
                result = shell.run_cell(cell.source, store_history=True)
            except BaseException as error:
                failure = error
            finally:
                timer.cancel()
    finally:
        timer.cancel()
        hook.write_output_prompt, hook.write_format_data = old_prompt, old_format

    error = failure or (
        (result.error_before_exec or result.error_in_exec) if result is not None else None
    )
    if timed_out.is_set():
        error = TimeoutError(f"Cell exceeded the {timeout}s timeout.")
    if error is not None:
        outputs.append(
            nbformat.v4.new_output(
                "error",
                ename=type(error).__name__,
                evalue=str(error),
                traceback=traceback.format_exception(type(error), error, error.__traceback__),
            )
        )
        raise CellFailed(f"{type(error).__name__}: {error}") from error


def execute_in_process(notebook, path: Path, timeout: int) -> None:
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    # An inline backend renders charts into notebook outputs without GUI windows.
    try:
        import matplotlib
        from matplotlib_inline.backend_inline import configure_inline_support

        matplotlib.use("module://matplotlib_inline.backend_inline")
        configure_inline_support(shell, "inline")
    except ImportError:
        pass
    exec(TEXT_PROGRESS_SETUP, shell.user_ns)

    for index, cell in enumerate(notebook.cells):
        start = report_cell(index, len(notebook.cells), cell)
        try:
            if cell.cell_type == "code" and cell.source.strip():
                run_cell_in_process(shell, cell, timeout)
        finally:
            save_notebook(notebook, path)
        print(f"  saved ({time.monotonic() - start:.1f}s)", flush=True)


def execute_with_nbclient(notebook, path: Path, timeout: int) -> None:
    from nbclient import NotebookClient

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
        force_raise_errors=True,
    )
    client.reset_execution_trackers()
    with client.setup_kernel(cwd=str(path.parent)):
        # Prepare text progress without adding an execution cell to the notebook.
        bootstrap = nbformat.v4.new_code_cell(TEXT_PROGRESS_SETUP)
        bootstrap_index = len(notebook.cells)
        notebook.cells.append(bootstrap)
        try:
            client.execute_cell(bootstrap, bootstrap_index, store_history=False)
        finally:
            notebook.cells.pop()
        client.code_cells_executed = 0
        for index, cell in enumerate(notebook.cells):
            start = report_cell(index, len(notebook.cells), cell)
            try:
                client.execute_cell(
                    cell, index, execution_count=client.code_cells_executed + 1
                )
            finally:
                save_notebook(notebook, path)
            print(f"  saved ({time.monotonic() - start:.1f}s)", flush=True)


@contextlib.contextmanager
def notebook_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        yield
    finally:
        os.chdir(previous)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook", nargs="?", type=Path, default=Path("dog_breed_classifier.ipynb")
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--in-process", action="store_true", help="IPython execution without sockets (default)"
    )
    mode.add_argument("--nbclient", action="store_true", help="Use an isolated Jupyter kernel")
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Timeout per code cell in seconds (default: 3600)"
    )
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")
    path = args.notebook.expanduser().resolve(strict=True)
    notebook = nbformat.read(path, as_version=4)
    # Clear old results before running so an interrupted run never leaves stale
    # later-cell predictions next to freshly executed earlier cells.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    save_notebook(notebook, path)
    executor = execute_with_nbclient if args.nbclient else execute_in_process
    print(f"Executing {path.name} ({'nbclient' if args.nbclient else 'in-process'})", flush=True)
    if not args.nbclient:
        print("Timeout interrupts Python; native operations may stop only when they return.", flush=True)
    try:
        with notebook_directory(path):
            executor(notebook, path, args.timeout)
    except BaseException as error:
        save_notebook(notebook, path)
        print(f"Stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        print(f"Completed cells and error outputs saved to {path.name}.", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(f"Finished. Saved {path.name}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
