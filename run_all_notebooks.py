from pathlib import Path
import argparse
import base64
import json
import os
from datetime import datetime
import nbformat
from nbclient import NotebookClient

SKIP_NOTEBOOKS = {
    "update_3_patchsize_project4.ipynb",
}


def ensure_gpu_environment():
    import torch

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA tidak tersedia. Pastikan PyTorch CUDA terinstal dan GPU terdeteksi.")

    print("GPU detected:", torch.cuda.get_device_name(0))
    return torch.device("cuda")


def prepend_gpu_guard(nb, cwd: Path):
    guard_cell = nbformat.v4.new_code_cell(
        """import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if not torch.cuda.is_available():
    raise RuntimeError('GPU tidak tersedia untuk eksekusi notebook ini')

device = torch.device('cuda')
print('Using device:', device)
print('GPU:', torch.cuda.get_device_name(0))
"""
    )
    nb.cells = [guard_cell] + nb.cells


def collect_output_text(outputs):
    text_chunks = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            text_chunks.extend(output.get("text", []))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/plain" in data:
                text_chunks.extend(data["text/plain"] if isinstance(data["text/plain"], list) else [data["text/plain"]])
            if "text/html" in data:
                text_chunks.append("[HTML output]")
            if "image/png" in data:
                text_chunks.append("[Image output saved to file]")
        elif output_type == "error":
            text_chunks.extend(output.get("traceback", []))
    return "".join(text_chunks)


def save_notebook_artifacts(nb, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    executed_notebook_path = run_dir / "executed_notebook.ipynb"
    with executed_notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    cell_summaries = []
    image_counter = 0
    all_text = []
    for idx, cell in enumerate(nb.cells, start=1):
        outputs = cell.get("outputs", [])
        output_text = collect_output_text(outputs)
        all_text.append(f"===== Cell {idx} ({cell.cell_type}) =====\n{output_text}\n")

        cell_summary = {
            "cell_index": idx,
            "cell_type": cell.cell_type,
            "output_text": output_text,
        }

        for output in outputs:
            if output.get("output_type") in {"display_data", "execute_result"}:
                data = output.get("data", {})
                if "image/png" in data:
                    image_counter += 1
                    image_path = images_dir / f"cell_{idx}_image_{image_counter}.png"
                    image_bytes = base64.b64decode(data["image/png"])
                    with image_path.open("wb") as fh:
                        fh.write(image_bytes)
                    cell_summary["saved_image"] = image_path.name

        cell_summaries.append(cell_summary)

    with (run_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    with (run_dir / "cell_outputs.json").open("w", encoding="utf-8") as f:
        json.dump(cell_summaries, f, indent=2)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "image_count": image_counter,
        }, f, indent=2)


def notebook_already_completed(run_dir: Path, notebook_path: Path) -> bool:
    completed_path = run_dir / notebook_path.stem / "executed_notebook.ipynb"
    return completed_path.exists()


def find_latest_run_index(output_root: Path) -> int | None:
    max_index = None
    for child in output_root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                index = int(child.name.split("run_")[1])
                if max_index is None or index > max_index:
                    max_index = index
            except ValueError:
                continue
    return max_index


def is_run_complete(run_dir: Path, notebooks: list[Path]) -> bool:
    return all(notebook_already_completed(run_dir, nb) for nb in notebooks)


def execute_notebook(notebook_path: Path, cwd: Path, timeout: int = 3600, output_root: Path | None = None, run_index: int = 1):
    print(f"\n=== Running {notebook_path.name} ===")
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    prepend_gpu_guard(nb, cwd)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(cwd)}}
    )

    os.chdir(cwd)
    client.execute()

    if output_root is not None:
        run_dir = output_root / f"run_{run_index:02d}" / notebook_path.stem
        save_notebook_artifacts(nb, run_dir)
        print(f"Saved artifacts to: {run_dir}")

    print(f"Completed: {notebook_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all notebooks in this workspace sequentially")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per notebook in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Only list notebooks without executing")
    parser.add_argument("--include-tensorflow", action="store_true", help="Also run notebooks that still depend on TensorFlow")
    parser.add_argument("--runs", type=int, default=1, help="How many times to run the notebooks")
    parser.add_argument("--output-dir", type=str, default="results", help="Folder where outputs from each run will be saved")
    parser.add_argument("--resume", action="store_true", help="Automatically resume the latest incomplete run or start a new run if the latest is complete")
    parser.add_argument("--run-index", type=int, default=None, help="Explicit start run index (overrides resume auto-detection)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    notebooks = [
        base_dir / "Ashoka_With_Hybrid_CNN_and_C_Swin.ipynb",
        base_dir / "DilatedSEDenseNet_WithHoldOut.ipynb",
        base_dir / "percobaan 3 hold out 224 patch 16 supis.ipynb",
        base_dir / "percobaan 3-2 hold out 224 patch 32 supis.ipynb",
        base_dir / "percobaan 3-3 hold out 224 patch 8 supis.ipynb",
    ]
    output_root = base_dir / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    print("Workspace:", base_dir)
    print("Notebooks to run:")
    for nb in notebooks:
        if nb.name in SKIP_NOTEBOOKS and not args.include_tensorflow:
            print(f"- {nb.name} (skipped for now)")
        else:
            print("-", nb.name)

    ensure_gpu_environment()

    if args.dry_run:
        print("\nDry run only. No notebooks executed.")
        raise SystemExit(0)

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    if args.resume and args.run_index is not None:
        print("Warning: --run-index provided together with --resume; explicit run index will override resume auto-detection.")

    if args.run_index is not None:
        start_run = args.run_index
    elif args.resume:
        latest_run = find_latest_run_index(output_root)
        if latest_run is None:
            start_run = 1
        else:
            latest_dir = output_root / f"run_{latest_run:02d}"
            start_run = latest_run if not is_run_complete(latest_dir, notebooks) else latest_run + 1
    else:
        start_run = 1

    start_run = max(start_run, 1)
    end_run = start_run + args.runs - 1

    print(f"Starting run sequence from run_{start_run:02d} to run_{end_run:02d}")

    for run_index in range(start_run, end_run + 1):
        print(f"\n===== RUN {run_index}/{end_run} =====")
        run_dir = output_root / f"run_{run_index:02d}"
        for nb in notebooks:
            if not nb.exists():
                raise FileNotFoundError(f"Notebook not found: {nb}")
            if nb.name in SKIP_NOTEBOOKS and not args.include_tensorflow:
                print(f"\nSkipping {nb.name} (TensorFlow notebook)")
                continue
            if notebook_already_completed(run_dir, nb):
                print(f"\nSkipping {nb.name} because it already has saved results in {run_dir / nb.stem}")
                continue
            execute_notebook(nb, base_dir, timeout=args.timeout, output_root=output_root, run_index=run_index)

    print(f"\nAll runnable notebooks completed successfully. Results saved under {output_root}")
