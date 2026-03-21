"""
build_labels.py
Build fraud labels for Medicaid providers from the LEIE exclusion dataset.
Output: npi, label (1 = NPI appears in LEIE with a non-zero/non-null NPI, 0 = not excluded).
All distinct billing provider NPIs from the Medicaid CSV receive a row; label defaults to 0.
"""
import argparse
import sys
from pathlib import Path

import duckdb


LEIE_REQUIRED_COLUMNS = {"NPI", "EXCLDATE"}
MEDICAID_REQUIRED_COLUMNS = {"BILLING_PROVIDER_NPI_NUM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fraud labels: one row per provider with npi and label (1=LEIE excluded, 0=not excluded)."
    )
    parser.add_argument("--leie_csv", required=True, help="Path to LEIE exclusion CSV.")
    parser.add_argument("--medicaid_csv", required=True, help="Path to raw Medicaid billing CSV.")
    parser.add_argument("--output_csv", required=True, help="Path to write output CSV.")
    return parser.parse_args()


def validate_columns(
    con: duckdb.DuckDBPyConnection, csv_path: str, required: set[str], label: str
) -> None:
    preview_df = con.execute(
        "SELECT * FROM read_csv_auto(?, SAMPLE_SIZE=10000) LIMIT 0", [csv_path]
    ).fetchdf()
    missing = required.difference(preview_df.columns)
    if missing:
        sys.stderr.write(f"{label} CSV missing columns: {', '.join(sorted(missing))}\n")
        raise SystemExit(1)


def run(leie_csv: str, medicaid_csv: str, output_csv: str) -> None:
    con = duckdb.connect()
    try:
        con.execute("PRAGMA progress_bar = true;")
    except Exception:
        pass

    for path in (leie_csv, medicaid_csv):
        if not Path(path).is_file():
            sys.stderr.write(f"Input not found: {path}\n")
            raise SystemExit(1)

    validate_columns(con, leie_csv, LEIE_REQUIRED_COLUMNS, "LEIE")
    validate_columns(con, medicaid_csv, MEDICAID_REQUIRED_COLUMNS, "medicaid")

    out_sql = f"""
        WITH medicaid_npis AS (
            SELECT DISTINCT CAST(BILLING_PROVIDER_NPI_NUM AS VARCHAR) AS npi
            FROM read_csv_auto('{medicaid_csv}')
        ),
        leie_npis AS (
            SELECT CAST(NPI AS VARCHAR) AS npi, MIN(CAST(EXCLDATE AS VARCHAR)) AS excldate
            FROM read_csv_auto('{leie_csv}')
            WHERE NPI IS NOT NULL
              AND TRIM(CAST(NPI AS VARCHAR)) <> ''
              AND CAST(NPI AS BIGINT) <> 0
            GROUP BY NPI
        )
        SELECT
            m.npi,
            CASE WHEN l.npi IS NOT NULL THEN 1 ELSE 0 END AS label,
            l.excldate
        FROM medicaid_npis m
        LEFT JOIN leie_npis l
            ON m.npi = l.npi
    """

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({out_sql}) TO ? (HEADER, DELIMITER ',')", [str(out_path)])
    print(f"Wrote: {out_path}")


def main() -> None:
    args = parse_args()
    run(args.leie_csv, args.medicaid_csv, args.output_csv)


if __name__ == "__main__":
    main()
