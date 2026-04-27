"""Extract key tables and numerical evidence from large arXiv MCP downloads."""
import json
import re
import sys
from pathlib import Path

CACHE = Path("C:/Users/HOME/.claude/projects/C--Users-HOME-JW-Research-Docs-FL-Peak-Project/5bcceb7c-b65e-4047-a27a-ebac60ee0dc4/tool-results")

PAPERS = {
    "BuildingsBench (2307.00142)": "mcp-arxiv-download_paper-1777215699789.txt",
    "Privacy-FL (2111.09248)": "mcp-arxiv-download_paper-1777215951212.txt",
    "PFL Smart Meter (2502.17226)": "mcp-arxiv-download_paper-1777215959994.txt",
}

KEYWORDS = [
    r"zero-shot",
    r"transfer learning",
    r"RMSE",
    r"MAPE",
    r"NRMSE",
    r"MAE",
    r"peak",
    r"individual",
    r"household",
    r"federated",
    r"FedAvg",
    r"personalized",
    r"benchmark",
    r"meta-learning",
    r"differential privacy",
    r"secure aggregation",
    r"residential",
    r"day-ahead",
    r"hit rate",
    r"convergence",
]


def extract_text(path):
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return " ".join(item.get("text", "") for item in data if isinstance(item, dict))
    except Exception:
        pass
    return raw


def find_context(text, keyword, n_chars=200):
    """Return list of (start_idx, snippet) for each match."""
    matches = []
    for m in re.finditer(rf"\b{keyword}\b", text, re.IGNORECASE):
        s = max(0, m.start() - n_chars)
        e = min(len(text), m.end() + n_chars)
        snippet = text[s:e].replace("\n", " ")
        matches.append((m.start(), snippet))
    return matches


def find_tables(text):
    """Heuristic: find lines that look like table rows (lots of numbers)."""
    lines = text.split("\n")
    rows = []
    for i, line in enumerate(lines):
        nums = re.findall(r"\d+\.\d+", line)
        if len(nums) >= 4:
            rows.append((i, line.strip()))
    return rows


def find_table_section(text, anchor, span=2500):
    """Get a long context around a table-anchor keyword."""
    matches = list(re.finditer(anchor, text, re.IGNORECASE))
    out = []
    for m in matches:
        s = max(0, m.start() - 100)
        e = min(len(text), m.end() + span)
        out.append((m.start(), text[s:e].replace("\n", " ")))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for label, fname in PAPERS.items():
        path = CACHE / fname
        if not path.exists():
            print(f"\n[MISS] {label}: {path}")
            continue
        print(f"\n{'='*80}\n# {label}\n{'='*80}")
        text = extract_text(path)

        if "BuildingsBench" in label:
            print("\n## Section: Zero-shot results (Table 3 anchor)")
            for idx, snip in find_table_section(text, r"Zero-shot STLF results", span=3000)[:1]:
                print(f"  ...{snip}...")
            print("\n## Section: power-law diminishing returns")
            for idx, snip in find_table_section(text, r"power-law", span=600)[:2]:
                print(f"  ...{snip}...")
            print("\n## Section: Transformer-L on Sceaux residential")
            for idx, snip in find_table_section(text, r"Sceaux", span=1500)[:2]:
                print(f"  ...{snip}...")
        elif "Privacy-FL" in label:
            print("\n## Section: scenarios comparison")
            for idx, snip in find_table_section(text, r"Scenario A", span=2500)[:1]:
                print(f"  ...{snip}...")
            print("\n## Section: differential privacy results")
            for idx, snip in find_table_section(text, r"differential privacy", span=1500)[:2]:
                print(f"  ...{snip}...")
            print("\n## Section: best architecture MAPE")
            for idx, snip in find_table_section(text, r"6\.7", span=400)[:2]:
                print(f"  ...{snip}...")
        elif "PFL" in label:
            print("\n## Section: Table III conventional vs LSTM")
            for idx, snip in find_table_section(text, r"Table\s*III", span=2500)[:1]:
                print(f"  ...{snip}...")
            print("\n## Section: meta-learning effect")
            for idx, snip in find_table_section(text, r"meta-learning", span=1000)[:2]:
                print(f"  ...{snip}...")
            print("\n## Section: ARIMA exponential comparison")
            for idx, snip in find_table_section(text, r"exponential smoothing", span=1500)[:1]:
                print(f"  ...{snip}...")


if __name__ == "__main__":
    main()
