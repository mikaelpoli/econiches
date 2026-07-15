import argparse
import pandas as pd
import econiches.utils as utils
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import econiches.utils as utils


def _extract_header(filepath):
    with open(filepath, "rt") as f:
        for line in f:
            if line.startswith("#") and not line.startswith("##"):
                header = line.strip().lstrip("#").split("\t")
                return header
  

def process_mag(args):
    mag_id, header, usecols, mag_files, annotations_dir = args
    filename = next((f for f in mag_files if f.startswith(mag_id)), None)
    filepath = annotations_dir / filename

    df = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        names=header,
        usecols=usecols,
        na_values="-"
    )

    coverage = df.notna().mean() * 100
    coverage["genome_size"] = int(len(df))
    coverage["mag_id"] = mag_id
    return coverage


# =============================
# MAIN PROGRAM
# =============================
if __name__ == "__main__":
    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process MAG annotation files and export coverage CSV.")

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="coverage.csv",
        help="Output CSV file name (default: results.csv)"
        )
    parser.add_argument(
        "--m", "--mags",
        type=str,
        default="None",
        help="A .csv file containing the IDs of the MAGs to analyze"
    )
    
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=8,
        help="Number of cores to use to compute coverage (default: 8)"
    )
    
    args = parser.parse_args()

    # Set globals
    BASE_DIR          = utils.set_base_directory()
    SRC_DIR           = BASE_DIR / 'src'
    utils.append_wd(SRC_DIR)
    LABELS_DIR        = BASE_DIR / 'data'
    ANNOTATIONS_DIR   = BASE_DIR.resolve().parent.parent / 'sofia' / 'eggnog_diamond_def' / 'annot'
    OUTPUT_DIR        = BASE_DIR / 'coverage'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LABELS_FILE       = LABELS_DIR / 'mag_labels.csv'
    LABELS_USECOLS    = ["mag_id", "metagenome_id", "environment"] #["mags", "Macro environment", "Environment"]

    FEATURES_SET      = set(["COG_category", "CAZy", "EC", "KEGG_ko", "PFAMs"])
    MAG_FILES         = {p.name for p in ANNOTATIONS_DIR.iterdir()}

    HEADER            = _extract_header(ANNOTATIONS_DIR / list(MAG_FILES)[0])

    OUTPUT_FILE       = OUTPUT_DIR / Path(args.output)

    N_CORES           = args.cores

    # -----------------------------
    # DATA IMPORT
    # -----------------------------
    labels = pd.read_csv(LABELS_FILE, sep=',', usecols=LABELS_USECOLS)

    if args.mags == "None":
        mag_list = sorted(labels["mags"].unique())
    else:
        mag_list = pd.read_csv(args.mags, sep=',', usecols="mag_id")
        mag_list = sorted(mag_list["mag_id"].unique())
    
    n = len(mag_list)

    # -----------------------------
    # COMPUTE COVERAGE
    # -----------------------------
    rows = []

    with Pool(N_CORES) as pool:
        print(f"Computing coverage using {N_CORES} cores. Output file: {OUTPUT_FILE}")
        args = [(mag, HEADER, FEATURES_SET, MAG_FILES, ANNOTATIONS_DIR) for mag in mag_list]
        for coverage in tqdm(pool.imap_unordered(process_mag, args), total=n, desc="Processing MAGs"):
            rows.append(coverage)

    col_order = ["mag_id", "genome_size"] + sorted(FEATURES_SET)
    results = pd.DataFrame(rows)
    results = results[col_order]
    results.to_csv(OUTPUT_FILE, float_format="%.3f", index=False)
    print(f"Results saved to {OUTPUT_FILE}")