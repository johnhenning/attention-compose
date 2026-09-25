"""Execute the demo in a fresh kernel; optionally save outputs for sharing."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the executed notebook to this path")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "attention_compose_demo.ipynb"
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(source.parent)}},
    )
    client.execute()
    nbformat.validate(notebook)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {count} code cells successfully, including all demo assertions.")


if __name__ == "__main__":
    main()
