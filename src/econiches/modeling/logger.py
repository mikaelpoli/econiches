from datetime import datetime
import logging
import os
from pathlib import Path
import sys

from econiches.utils import print_section_separator

def get_logger(name: str, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def ensure_logger(log=None):
    if log is not None:
        return log

    logger = logging.getLogger("console_fallback")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


if __name__ == "__main__":
    import econiches.modeling.paths as paths

    args = paths.parse_args()
    proj_paths = paths.Paths(env=args.env,
                             root_dir=args.root.resolve(),
                             model_type=args.mod,
                             annot_family=args.fam,
                             log_run_dir="run_demo")
    paths.ensure_dirs(proj_paths)
    paths.print_paths(proj_paths)

    print_section_separator()

    script_name = os.path.basename(__file__)
    logger_name = f"econiches_{script_name}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = get_logger(name=logger_name, log_file=proj_paths.logs / f"{run_id}.log")
    print(f"Initialized logger: {logger}")

    logger = ensure_logger(log=logger)
    logger_fallback = ensure_logger()
    print(f"Initialized fallback logger: {logger_fallback}")