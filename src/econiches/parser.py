# ==========================================================
# Performs:
#     * Parsing of each .emapper.annotation file
# ==========================================================
# Produces:
#     * Parsed eggNOG-mapper v2 MAG annotations (format: parquet)
#       containing the per-MAG raw counts of all terms in:
#       - EC
#       - COG_category
#       - KEGG_ko
#       - CAZy
#       - PFAM
# ==========================================================

from multiprocessing import Pool
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re
from tqdm import tqdm
import econiches.utils as utils


# -----------------------------
# EXTRACTORS
# -----------------------------
class FeatureExtractor:
    def __init__(self, column_name, prefix):
        self.prefix = prefix
        self.column_name = column_name
        self.col_idx = None
        self.counts = {}

    def set_column_index(self, header):
        self.col_idx = header.index(self.column_name)

    def process(self, row):
        raise NotImplementedError

    def finalize(self):
        return {f"{self.prefix}_{k}": v for k, v in self.counts.items()}


class TokenCountExtractor(FeatureExtractor):
    """
    Generic extractor for delimiter-separated annotation tokens.
    """
    pattern = re.compile(r"[;,|]")

    def __init__(self, column_name, prefix):
        super().__init__(column_name, prefix)

    def validate_token(self, token):
        return True

    def normalize_token(self, token):
        return token

    def process(self, row):
        field = row[self.col_idx]

        if not field or field == "-":
            return

        tokens = self.pattern.split(field)

        for token in tokens:
            token = token.strip()

            if not token:
                continue

            if not self.validate_token(token):
                continue

            key = self.normalize_token(token)

            if not key:
                continue

            self.counts[key] = self.counts.get(key, 0) + 1
    

class COGExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("COG_category", prefix="COG")

    def process(self, row):
        field = row[self.col_idx]

        if not field or field == "-":
            return

        for letter in field:
            if letter.isalpha() and letter.isupper():
                self.counts[letter] = self.counts.get(letter, 0) + 1


# -----------------------------
# PARSING PIPELINE
# -----------------------------
def parse_annotation(filepath, extractors, features_set):
    total_genes = 0

    with open(filepath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                header = line.strip().lstrip("#").split("\t")

                if features_set.issubset(header):
                    for extractor in extractors:
                        extractor.set_column_index(header)
                    continue

            if line.startswith("#"): # Skip metadata 
                continue

            row = line.rstrip("\n").split("\t")
            total_genes += 1

            for extractor in extractors:
                extractor.process(row)

    features = {}
    for extractor in extractors:
        features.update(extractor.finalize())
    features.update({"genome_size": total_genes})

    return features


def parse_mag(args):
    mag_id, annotations_dir, features_set, files = args

    filename = next((f for f in files if f.startswith(mag_id)), None)
    filepath = annotations_dir / filename

    extractors = [
        COGExtractor(),
        TokenCountExtractor("CAZy", "CAZy"),
        TokenCountExtractor("EC", "EC"),
        TokenCountExtractor("PFAMs", "PFAM"),
        TokenCountExtractor("KEGG_ko", "KO")
    ]
    
    features = parse_annotation(filepath, extractors, features_set)

    return mag_id, features


def parse_taxonomy(labels):
    labels.rename(columns={
        "Environment": "environment",
        "Macro environment": "macro_environment",
        "Sub environment": "sub_environment"
        }, inplace=True)
    
    tax_split = labels["joined"].str.split(";", expand=True)
    tax_split.columns = [
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species"
    ]

    for col in tax_split.columns:
        tax_split[col] = tax_split[col].str.replace(r"^[a-z]__", "", regex=True)

    tax_split = tax_split.replace("", pd.NA)
    labels.drop("joined", axis="columns", inplace=True)

    return pd.concat([labels, tax_split], axis=1)


# -----------------------------
# OUTPUT
# -----------------------------
def save_parquet(features_dict, output_file):
    print(f"Writing to parquet...")

    annotations = (
        pd.DataFrame.from_dict(features_dict, orient="index")
        .fillna(0)
        .astype("int16")
    )
    
    table = pa.Table.from_pandas(annotations, preserve_index=True)

    pq.write_table(
        table,
        output_file,
        compression="snappy",
        use_dictionary=False,
        write_statistics=False
    )

    print(f"Saved {output_file}")


# =============================
# MAIN PROGRAM
# =============================
if __name__ == "__main__":
    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    BASE_DIR          = utils.set_base_directory()
    SRC_DIR           = BASE_DIR / 'src'
    utils.append_wd(SRC_DIR)
    LABELS_DIR        = BASE_DIR / 'data'
    ANNOTATIONS_DIR   = BASE_DIR.resolve().parent.parent / 'sofia' / 'eggnog_diamond_def' / 'annot'
    OUTPUT_DIR        = BASE_DIR / 'parsed'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LABELS_FILE       = LABELS_DIR / 'mag_labels.csv'
    LABELS_USECOLS    = ["mags", "Macro environment", "Environment", "Sub environment", "GA", "OTU", "joined"]

    FEATURES_SET      = set(["COG_category", "CAZy", "EC", "KEGG_ko", "PFAMs"])
    MAG_FILES         = {p.name for p in ANNOTATIONS_DIR.iterdir()}

    OUTPUT_FILE_ANNOT = OUTPUT_DIR / "mag_features.parquet"
    OUTPUT_FILE_LABS  = OUTPUT_DIR / "mag_labels.csv"

    N_CORES           = 8
    PARSE             = False

    # -----------------------------
    # DATA IMPORT
    # -----------------------------
    labels = pd.read_csv(LABELS_FILE, sep=',', usecols=LABELS_USECOLS)

    mag_list = sorted(labels["mags"].unique())
    n = len(mag_list)

    print(f"Unique MAGs: {n}\n")

    # -----------------------------
    # Parsing Pipeline
    # -----------------------------
    # Labels
    labels_parsed = parse_taxonomy(labels)
    labels_parsed[["metagenome_id", "sequential_id"]] = (
        labels_parsed["mags"].str.split(pat="_", n=1, expand=True).values
    )
    labels_parsed[["sequential_id", "file_extension"]] = (
        labels_parsed["sequential_id"].str.split(pat=".", n=1, expand=True).values
    )
    labels_parsed.to_csv(OUTPUT_FILE_LABS, index=False)

    # Annotations
    if PARSE:

        exists = 0
        features_dict = {}

        with Pool(N_CORES) as pool:
            args = [(mag, ANNOTATIONS_DIR, FEATURES_SET, MAG_FILES) for mag in mag_list]
            results = []
            for result in tqdm(pool.imap_unordered(parse_mag, args), total=n):
                results.append(result)

        for mag_id, features in results:
            if features is not None:
                exists += 1
                features_dict[mag_id] = features

        print(f"Found {exists} annotation files.")
        print(f"Missing {n - exists} annotation files.")       
        
        save_parquet(features_dict, OUTPUT_FILE_ANNOT)

