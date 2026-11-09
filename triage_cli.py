"""Automatic comment triage from the command line (no Spark, no website needed).

One file:    python triage_cli.py comments.csv --out reports/auto
A whole folder (this is what the GitHub automation runs):
             python triage_cli.py --inbox inbox --out reports/auto
Each CSV needs a column named comment (or text / body / message / content).
"""
import argparse
import sys
from pathlib import Path

from src.reporting import build_report
from src.triage import read_rows, rows_to_csv, triage_rows


def process(path: Path, out_dir: Path) -> None:
    fields, rows, text_col = read_rows(path.read_text(encoding="utf-8", errors="replace"))
    result = triage_rows(rows, text_col)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{path.stem}_triaged.csv").write_text(rows_to_csv(fields, result), encoding="utf-8")
    (out_dir / f"{path.stem}_report.md").write_text(
        build_report(result, text_col, title=f"Comment report: {path.name}"), encoding="utf-8")
    print(f"{path.name}: {len(result)} comments -> {out_dir}/{path.stem}_triaged.csv and _report.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="CSV file(s) of comments")
    parser.add_argument("--inbox", type=Path, help="process every .csv in this folder")
    parser.add_argument("--out", type=Path, default=Path("reports/auto"), help="output folder")
    args = parser.parse_args()

    files = list(args.files)
    if args.inbox:
        files += sorted(args.inbox.glob("*.csv"))
    if not files:
        print("No CSV files found.")
        return 0
    for path in files:
        process(path, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
