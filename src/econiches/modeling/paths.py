from pathlib import Path
from tabulate import tabulate
import argparse

IGNORE = {".git", "__pycache__"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        choices=["environment", "macro_environment"],
        default="environment",
        help="Environment labels to use"
    )
    parser.add_argument(
        "--fam",
        type=str,
        choices=["cog", "ec", "pfam", "cazy", "ko", "all", "none"],
        default="all",
        help="Type of functional annotations to use as features"
    )
    parser.add_argument(
        "--mod",
        type=str,
        choices=["rf", "lr", "ensemble"],
        default="lr",
        help="Type of ML model"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default="./",
        help="Set root project directory"
    )
    return parser.parse_args()


class Paths:
    def __init__(
            self,
            env: str,
            root_dir: Path,
            model_type: str,
            annot_family: str,
            log_run_dir: str
        ):
        self.cwd = Path().resolve()
        self.root = root_dir
        self.env = env

        self.train = self.root / "filtered" / "split" / env / "train_test" / "train"
        self.test = self.root / "filtered" / "split" / env / "train_test" / "test"
        self.full = self.root / "filtered" / "full"

        if model_type in ["benchmark", "ensemble"]:
            self.logs = self.root / "logs" / env / model_type / log_run_dir
            self.plots = self.root / "plots" / "modeling" / env / model_type
        else:
            self.logs = self.root / "logs" / env / model_type / annot_family / log_run_dir
            self.plots = self.logs / "plots"
        
        if model_type in ["rf", "lr"]:
            model_dir = "main"
            config_filename = f"config_{model_type}.yaml"
        else:
            model_dir = model_type
            config_filename = "config.yaml"
        
        self.config_in = self.root / "models" / model_dir

        self.files = {
            "X_train": self.train / "X_train.parquet",
            "y_train": self.train / "y_train.csv",
            "X_test" : self.test / "X_test.parquet",
            "y_test" : self.test / "y_test.csv",
            "full_features": self.full / "features.parquet",
            "full_labels": self.full / "labels.csv",
            "config": self.config_in / config_filename
        }


def ensure_dirs(paths: Paths):
    dirs = [
        paths.train,
        paths.test,
        paths.full,
        paths.logs,
        paths.config_in,
        paths.plots
    ]

    exists = 0
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        exists += 1

    if exists < len(dirs):
        print("One or more directories don't exist")
        exit(1)


def print_paths(project_paths):
    print("--- PATHS ---")

    rows = []
    for key, value in project_paths.__dict__.items():
        if key not in ["files", "env"]:
            rows.append((key.title(), value))

    print(tabulate(rows, headers=["Name", "Path"]))

    print("\n--- FILES ---")

    file_rows = [(key, value) for key, value in project_paths.files.items()]
    print(tabulate(file_rows, headers=["File", "Path"]))


if __name__ == "__main__":
    args = parse_args()
    proj_paths = Paths(env=args.env,
                       root_dir=args.root.resolve(),
                       model_type=args.mod,
                       annot_family=args.fam,
                       log_run_dir="run_demo")
    ensure_dirs(proj_paths)
    print_paths(proj_paths)

    exit(0)