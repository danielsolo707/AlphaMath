"""Download AI-MO AIME validation set and convert to ALPHA-MATH format."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "benchmarks" / "aime"


def main() -> None:
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    problems: list[dict] = []
    bad: list = []
    for i, row in enumerate(ds):
        ans_raw = row["answer"]
        try:
            if isinstance(ans_raw, float) and ans_raw.is_integer():
                ans = int(ans_raw)
            else:
                ans = int(str(ans_raw).strip())
        except Exception as exc:  # noqa: BLE001
            bad.append((row.get("id"), ans_raw, str(exc)))
            continue
        if not 0 <= ans <= 999:
            bad.append((row.get("id"), ans, "out_of_range"))
            continue

        url = str(row.get("url") or "")
        year = "unknown"
        for y in range(2020, 2026):
            if str(y) in url:
                year = str(y)
                break

        orig = row.get("id", i)
        pid = f"aime_{year}_{int(orig):03d}" if str(orig).isdigit() and year != "unknown" else f"aime_{i:03d}"
        problems.append(
            {
                "id": pid,
                "source": "AI-MO/aimo-validation-aime",
                "difficulty": "hard",
                "problem": str(row["problem"]).strip(),
                "answer": ans,
                "tags": ["aime", year],
                "url": url,
                "original_id": orig,
            }
        )

    problems.sort(
        key=lambda p: int(p["original_id"]) if str(p.get("original_id", "")).isdigit() else 0
    )

    json_path = OUT_DIR / "aime_2022_2024.json"
    json_path.write_text(json.dumps(problems, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = OUT_DIR / "aime_2022_2024.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "answer", "source", "difficulty", "problem", "url"]
        )
        writer.writeheader()
        for item in problems:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})

    # Kaggle-friendly flat copy without solutions (already no solutions)
    kaggle_dir = ROOT / "kaggle" / "aime_benchmark_dataset"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    (kaggle_dir / "aime_2022_2024.json").write_text(
        json.dumps(problems, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = """# ALPHA-MATH AIME benchmark (private eval set)

Source: HuggingFace `AI-MO/aimo-validation-aime` (90 AIME-style problems, answers in 0-999).

Used only for labeled evaluation of ALPHA-MATH. Attach this dataset to the evaluation kernel.

Do not treat scores as an official AOPS leaderboard.
"""
    (kaggle_dir / "README.md").write_text(readme, encoding="utf-8")

    meta = {
        "name": "aime_2022_2024_validation",
        "source_dataset": "AI-MO/aimo-validation-aime",
        "n_problems": len(problems),
        "answer_min": min(p["answer"] for p in problems),
        "answer_max": max(p["answer"] for p in problems),
        "years_in_urls": sorted({t for p in problems for t in p["tags"] if t.isdigit()}),
        "bad_rows": bad,
        "local_json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
        "local_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "kaggle_payload": str(kaggle_dir.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT_DIR / "DATASET_CARD.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
