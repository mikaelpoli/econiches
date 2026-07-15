from collections.abc import Mapping
import numpy as np
from tabulate import tabulate
from pathlib import Path
import sys
import yaml

def set_base_directory():
    try:
        base_dir = Path(__file__).resolve().parent.parent
    except NameError:
        base_dir = Path().resolve()
    return base_dir


def append_wd(wd):
    wd = str(wd)
    if wd not in sys.path:
        sys.path.append(wd)


def print_data_check_header():
    print("\n┌─────────────────┐")
    print("│   DATA CHECK    │")
    print("└─────────────────┘\n")


def print_section_separator():
    print("\n" + "=" * 30 + "\n")


def print_sample_array(
        array: np.ndarray,
        num_samples: int = 5,
        title: str = "Sample array",
        tablefmt: str = "simple"
    ):
    headers = [f"C{i}" for i in range(array.shape[1])]

    print(f"\n{title}:\n")

    print(
        tabulate(
            array[:num_samples], 
            headers=headers,
            tablefmt=tablefmt
        )
    )


def choose_run_id(base_path: Path) -> str:
    run_ids = sorted(
        [p.name for p in base_path.iterdir() if p.is_dir()],
        reverse=True  # newest first if timestamp-based
    )

    if not run_ids:
        raise ValueError(f"No run_ids found in {base_path}")

    print(f"Available runs in {base_path}:")
    for i, run in enumerate(run_ids, 1):
        print(f"{i}. {run}")

    choice = input("Select run (number or name; Enter = latest): ").strip()

    if choice == "":
        return run_ids[0]
    elif choice.isdigit():
        return run_ids[int(choice) - 1]
    elif choice in run_ids:
        return choice
    else:
        raise ValueError("Invalid selection")
        

def load_preds(filepath):
    loaded = np.load(filepath)
    return {
        "preds": loaded["preds"],
        "probs": loaded["probs"]
    }