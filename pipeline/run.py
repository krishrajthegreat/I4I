# -*- coding: utf-8 -*-
"""
pipeline/run.py
===============
Interactive end-to-end pipeline runner for the Machine Type Classifier dataset.

Walks you through Stages 2-5 one class at a time, asking for confirmation at
each stage boundary rather than running everything automatically.

Usage
-----
    .venv\\Scripts\\python.exe pipeline/run.py

Flow
----
    Step 1  : Class selection (validated against config.py + dataset/train/)
    Stage 2 : Scrape     — with progress milestones every 25 images
    Stage 3 : Clean      — pHash dedup + resolution filter
    Stage 4 : CLIP filter — optional 4b calibration for confusable classes
    Stage 5 : Diversity dedup + K-selection

No stage scripts are modified. This file imports their core functions directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root on sys.path so all pipeline.* imports work regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    DATASET_TRAIN_DIR,
    K_MIN_CONFUSABLE,
    K_MIN_STANDARD,
    REFERENCE_IMAGES_DIR,
    SCRAPE_CAP_PER_CLASS,
    class_by_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MILESTONE: int = 25          # print a progress line every N images (Stage 2)
EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Default CLIP thresholds (used if calibration is skipped)
DEFAULT_COARSE_THRESH: float = 0.18
DEFAULT_FINE_THRESH: float = 0.30


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "=", width: int = 68) -> str:
    return char * width


def _count_images(folder: str) -> int:
    d = DATASET_TRAIN_DIR / folder
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS)


def _prompt(msg: str) -> str:
    """Read non-empty input, re-prompting until the user types something."""
    while True:
        val = input(msg).strip()
        if val:
            return val
        print("  [!] Input cannot be empty — please try again.")


def _prompt_optional(msg: str, default: str = "") -> str:
    val = input(msg).strip()
    return val if val else default


def _confirm(question: str) -> bool:
    """Ask a y/N question. Returns True only for 'y' / 'yes'."""
    ans = _prompt_optional(f"\n{question} [y/N]: ", default="n").lower()
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Step 1 — Class selection
# ---------------------------------------------------------------------------

def ask_classes() -> list[dict]:
    """
    Prompt for class selection, validate against ALL_CLASS_NAMES, and return
    a list of class dicts (as defined in config.CLASSES).
    """
    print("\n" + _hr())
    print("STEP 1 -- CLASS SELECTION")
    print(_hr())
    print("  Available classes (images already in dataset/train/):")
    for name in ALL_CLASS_NAMES:
        count = _count_images(name)
        count_str = f"{count} images" if count else "not yet scraped"
        print(f"    * {name:<25} ({count_str})")
    print()

    while True:
        raw = _prompt(
            "  Which class(es) to process?\n"
            "  Enter a class name, comma-separated names, or 'all': "
        )

        if raw.strip().lower() == "all":
            print(f"\n  All {len(ALL_CLASS_NAMES)} classes selected:")
            for name in ALL_CLASS_NAMES:
                print(f"      {name}")
            if _confirm("  Confirm?"):
                return list(CLASSES)
            print("  Restarting class selection...\n")
            continue

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        invalid = [t for t in tokens if t not in ALL_CLASS_NAMES]
        if invalid:
            print(f"\n  [!] Unknown class(es): {invalid}")
            print(f"      Valid names: {ALL_CLASS_NAMES}\n")
            continue

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [t for t in tokens if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
        return [class_by_name(n) for n in unique]


# ---------------------------------------------------------------------------
# Step 2 — Scrape cap
# ---------------------------------------------------------------------------

def ask_scrape_cap() -> int:
    print("\n" + _hr())
    print("STEP 2 -- SCRAPE SETTINGS")
    print(_hr())
    while True:
        raw = _prompt_optional(
            f"  Max images to scrape per class? [default {SCRAPE_CAP_PER_CLASS}]: ",
            default=str(SCRAPE_CAP_PER_CLASS),
        )
        try:
            val = int(raw)
            if val > 0:
                return val
            print("  [!] Must be a positive integer.")
        except ValueError:
            print("  [!] Please enter a whole number.")


# ---------------------------------------------------------------------------
# Stage 2 — Scrape with milestone progress
# ---------------------------------------------------------------------------

def run_stage2(classes: list[dict], cap: int) -> dict[str, dict]:
    """
    Scrape images for each class. Returns {class_name: scrape_result_dict}.
    Monkey-patches stage2_scrape._download_image to inject milestone prints
    without modifying the stage script itself.
    """
    import pipeline.stage2_scrape as _s2
    from pipeline.stage2_scrape import scrape_class

    print("\n" + _hr())
    print("STAGE 2 -- SCRAPE")
    print(_hr())

    # --- Milestone wrapper ---
    _orig = _s2._download_image
    _prog: dict = {"count": 0, "cap": cap, "name": ""}

    def _milestone_download(url: str, dest: Path, timeout: int = _s2.DOWNLOAD_TIMEOUT) -> bool:
        result = _orig(url, dest, timeout)
        if result:
            _prog["count"] += 1
            if _prog["count"] % MILESTONE == 0:
                print(
                    f"  [{_prog['name']}] {_prog['count']}/{_prog['cap']} images scraped...",
                    flush=True,
                )
        return result

    results: dict[str, dict] = {}
    try:
        _s2._download_image = _milestone_download
        for cls in classes:
            name = cls["name"]
            _prog["count"] = 0
            _prog["name"] = name
            print(f"\n  [{name}] Starting scrape — target={cap}...")
            result = scrape_class(cls, resume=False, cap=cap)
            results[name] = result
            final = result.get("final_count", 0)
            downloaded = result.get("total_downloaded", 0)
            print(f"  [{name}] Done — {final} raw images on disk ({downloaded} new this run).")
    finally:
        _s2._download_image = _orig  # always restore, even if we crash

    return results


# ---------------------------------------------------------------------------
# Stage 3 — Clean
# ---------------------------------------------------------------------------

def run_stage3(classes: list[dict]) -> dict[str, dict]:
    """
    Run pHash dedup + resolution filter on each class.
    Returns {class_name: clean_result_dict}.
    """
    from pipeline.stage3_clean import clean_class

    print("\n" + _hr())
    print("STAGE 3 -- CLEAN (corrupt / low-res / pHash dedup)")
    print(_hr())

    results: dict[str, dict] = {}
    for cls in classes:
        name = cls["name"]
        print(f"\n  [{name}] Cleaning...", flush=True)
        res = clean_class(name)
        results[name] = res
        raw    = res["raw_count"]
        clean  = res["final_count"]
        drop   = raw - clean
        print(
            f"  [{name}] Done — {raw} raw -> {clean} clean "
            f"(removed: {res['corrupt_removed']} corrupt, "
            f"{res['low_res_removed']} low-res, "
            f"{res['duplicates_removed']} dupes)"
        )
        if clean == 0:
            print(f"  [{name}] WARNING: 0 images remain after cleaning.")

    return results


# ---------------------------------------------------------------------------
# Stage 4b — Calibration (optional, per-class)
# ---------------------------------------------------------------------------

def _run_calibration(cls_name: str, model, preprocess, tokenizer, device) -> tuple[float, float]:
    """
    Run threshold calibration for a single confusable class.
    Asks the user for good/bad sample dirs, runs calibrate(), then asks
    whether to use the recommended thresholds or enter custom ones.
    Returns (coarse_thresh, fine_thresh).
    """
    from pipeline.stage4b_calibrate import calibrate

    print(f"\n  -- Calibration for '{cls_name}' --")

    default_good = str(DATASET_TRAIN_DIR / cls_name)
    good_raw = _prompt_optional(
        f"    Good samples dir [default: {default_good}]: ",
        default=default_good,
    )
    good_dir = Path(good_raw)

    bad_raw = _prompt_optional(
        "    Bad samples dir (optional, press Enter to skip): ",
        default="",
    )
    bad_dir = Path(bad_raw) if bad_raw else None

    calibrate(cls_name, good_dir, bad_dir)

    use_rec = _prompt_optional(
        "    Use the recommended thresholds printed above? [Y/n]: ",
        default="y",
    ).lower()
    if use_rec in ("y", "yes", ""):
        # Re-run calibrate just to get the numbers back... or ask user to re-enter.
        # calibrate() prints but doesn't return values, so we ask user to type them in.
        print("    (Enter the recommended values printed above.)")
        coarse = float(_prompt("    Coarse threshold: "))
        fine   = float(_prompt("    Fine threshold  : "))
    else:
        coarse = float(_prompt_optional(f"    Coarse threshold [default {DEFAULT_COARSE_THRESH}]: ", str(DEFAULT_COARSE_THRESH)))
        fine   = float(_prompt_optional(f"    Fine threshold   [default {DEFAULT_FINE_THRESH}]: ",   str(DEFAULT_FINE_THRESH)))

    return coarse, fine


# ---------------------------------------------------------------------------
# Stage 4 — CLIP filter
# ---------------------------------------------------------------------------

def run_stage4(
    classes: list[dict],
    model,
    preprocess,
    tokenizer,
    device,
) -> dict[str, dict]:
    """
    Run hybrid CLIP filtering on each class.
    For confusable classes, optionally runs Stage 4b calibration first.
    Returns {class_name: filter_result_dict}.
    """
    from pipeline.stage4_clip_filter import filter_class

    print("\n" + _hr())
    print("STAGE 4 -- HYBRID CLIP FILTER")
    print(_hr())

    results: dict[str, dict] = {}
    for cls in classes:
        name          = cls["name"]
        is_confusable = cls["confusable"]
        coarse        = DEFAULT_COARSE_THRESH
        fine          = DEFAULT_FINE_THRESH

        print(f"\n  [{name}] {'confusable' if is_confusable else 'standard (coarse only)'}")

        if is_confusable:
            ref_dir   = REFERENCE_IMAGES_DIR / name
            has_refs  = ref_dir.exists() and any(
                f.suffix.lower() in EXTS for f in ref_dir.iterdir()
            )
            if has_refs:
                if _confirm(f"  Run threshold calibration (stage 4b) for '{name}' now?"):
                    coarse, fine = _run_calibration(name, model, preprocess, tokenizer, device)
            else:
                print(f"  [!] No reference images found in {ref_dir} — fine filter will be skipped.")

        print(f"  [{name}] Filtering with coarse={coarse}, fine={fine}...", flush=True)
        res = filter_class(name, model, preprocess, tokenizer, device, coarse, fine)
        results[name] = res

        accepted = res["final_accepted"]
        inp      = res["input_count"]
        rej_c    = res["rejected_coarse"]
        rej_f    = res["rejected_fine"]
        print(
            f"  [{name}] Done — {inp} input -> {accepted} accepted "
            f"(rejected: {rej_c} coarse, {rej_f} fine)"
        )
        if accepted == 0:
            print(f"  [{name}] WARNING: 0 images remain after CLIP filtering.")

    return results


# ---------------------------------------------------------------------------
# Stage 5 — Diversity dedup + K-selection
# ---------------------------------------------------------------------------

def run_stage5(
    classes: list[dict],
    model,
    preprocess,
    tokenizer,
    device,
) -> dict[str, dict]:
    """
    Run greedy diversity dedup and K-selection on each class.
    Returns {class_name: kselect_result_dict}.
    """
    from pipeline.stage5_diversity_kselect import process_class

    print("\n" + _hr())
    print("STAGE 5 -- DIVERSITY DEDUP + K-SELECTION")
    print(_hr())

    results: dict[str, dict] = {}
    for cls in classes:
        name = cls["name"]
        print(f"\n  [{name}] Running diversity dedup...", flush=True)
        res = process_class(name, model, preprocess, tokenizer, device)
        results[name] = res

        sel   = res["selected_count"]
        inp   = res["input_count"]
        floor = res["hard_floor"]
        rem   = res["removed_diversity"]
        flag  = res["flagged"]

        status = "FLAGGED" if flag else "PASS"
        print(
            f"  [{name}] Done — {inp} input -> {sel} selected "
            f"(removed {rem} near-dupes) | floor={floor} | {status}"
        )
        if flag:
            print(
                f"  [{name}] WARNING: {sel} images < required floor {floor}. "
                f"Consider a scrape top-up before proceeding."
            )

    return results


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(
    classes: list[dict],
    s2: dict[str, dict] | None,
    s3: dict[str, dict] | None,
    s4: dict[str, dict] | None,
    s5: dict[str, dict] | None,
) -> None:
    print("\n" + _hr())
    print("FINAL SUMMARY — This Run")
    print(_hr())

    col = max(len(cls["name"]) for cls in classes) + 2
    header = (
        f"  {'Class':<{col}} {'Raw':>6}  {'Clean':>6}  "
        f"{'Filtered':>8}  {'Selected':>8}  {'Floor':>5}  Status"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cls in classes:
        name          = cls["name"]
        is_confusable = cls["confusable"]
        floor         = K_MIN_CONFUSABLE if is_confusable else K_MIN_STANDARD

        raw      = s2[name].get("final_count",    "—") if s2 and name in s2 else "—"
        clean    = s3[name].get("final_count",    "—") if s3 and name in s3 else "—"
        filtered = s4[name].get("final_accepted", "—") if s4 and name in s4 else "—"
        selected = s5[name].get("selected_count", "—") if s5 and name in s5 else "—"

        # Determine status from furthest stage reached
        if s5 and name in s5:
            status = "FLAGGED" if s5[name]["flagged"] else "PASS"
        elif s4 and name in s4:
            status = "PASS" if s4[name]["final_accepted"] > 0 else "FLAGGED (0 images)"
        elif s3 and name in s3:
            status = "PASS" if s3[name]["final_count"] > 0 else "FLAGGED (0 images)"
        elif s2 and name in s2:
            status = "PASS" if s2[name].get("final_count", 0) > 0 else "FLAGGED (0 images)"
        else:
            status = "NOT RUN"

        def _fmt(v) -> str:
            return f"{v:>8}" if isinstance(v, int) else f"{'—':>8}"

        print(
            f"  {name:<{col}} {raw if isinstance(raw, str) else raw:>6}  "
            f"{clean if isinstance(clean, str) else clean:>6}  "
            f"{_fmt(filtered)}  {_fmt(selected)}  {floor:>5}  {status}"
        )

    print(_hr())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + _hr())
    print(" Machine Classifier -- Interactive Pipeline Runner")
    print(_hr())

    # ------------------------------------------------------------------
    # Step 1 — Class selection
    # ------------------------------------------------------------------
    classes = ask_classes()
    names   = [cls["name"] for cls in classes]

    # ------------------------------------------------------------------
    # Step 2 — Scrape
    # ------------------------------------------------------------------
    cap = ask_scrape_cap()
    s2_results = run_stage2(classes, cap)

    # Summarise Stage 2
    print("\n" + _hr("-"))
    print("  Stage 2 complete:")
    for name in names:
        r = s2_results[name]
        print(f"    {name:<25} {r.get('final_count', 0):>5} images on disk")
    print(_hr("-"))

    # ------------------------------------------------------------------
    # Checkpoint 2 -> 3
    # ------------------------------------------------------------------
    if not _confirm("Stage 2 complete. Continue to Stage 3 (Cleaning)?"):
        print("\nStopped after Stage 2.")
        print_final_summary(classes, s2_results, None, None, None)
        return

    # ------------------------------------------------------------------
    # Stage 3 — Remove classes that have 0 images
    # ------------------------------------------------------------------
    s3_results = run_stage3(classes)

    # Drop classes with 0 clean images from further stages
    classes_3 = [cls for cls in classes if s3_results[cls["name"]]["final_count"] > 0]
    stopped_at_3 = [cls for cls in classes if cls not in classes_3]
    if stopped_at_3:
        print("\n  The following classes had 0 images after Stage 3 and will not proceed:")
        for cls in stopped_at_3:
            print(f"    * {cls['name']}")

    if not classes_3:
        print("\nAll classes have 0 images after Stage 3. Stopping.")
        print_final_summary(classes, s2_results, s3_results, None, None)
        return

    print("\n" + _hr("-"))
    print("  Stage 3 complete:")
    for name in names:
        r = s3_results[name]
        print(f"    {name:<25} {r['final_count']:>5} clean images")
    print(_hr("-"))

    # ------------------------------------------------------------------
    # Checkpoint 3 -> 4
    # ------------------------------------------------------------------
    if not _confirm("Stage 3 complete. Continue to Stage 4 (Hybrid CLIP Filtering)?"):
        print("\nStopped after Stage 3.")
        print_final_summary(classes, s2_results, s3_results, None, None)
        return

    # ------------------------------------------------------------------
    # Load CLIP model — shared between Stage 4 and Stage 5
    # ------------------------------------------------------------------
    print("\n  Loading OpenCLIP model (shared for Stage 4 and Stage 5)...")
    from pipeline.stage4_clip_filter import load_model
    model, preprocess, tokenizer, device = load_model()

    # ------------------------------------------------------------------
    # Stage 4 — CLIP filter
    # ------------------------------------------------------------------
    s4_results = run_stage4(classes_3, model, preprocess, tokenizer, device)

    # Drop classes with 0 accepted images
    classes_4 = [cls for cls in classes_3 if s4_results[cls["name"]]["final_accepted"] > 0]
    stopped_at_4 = [cls for cls in classes_3 if cls not in classes_4]
    if stopped_at_4:
        print("\n  The following classes had 0 images after Stage 4 and will not proceed:")
        for cls in stopped_at_4:
            print(f"    * {cls['name']}")

    if not classes_4:
        print("\nAll classes have 0 images after Stage 4. Stopping.")
        print_final_summary(classes, s2_results, s3_results, s4_results, None)
        return

    print("\n" + _hr("-"))
    print("  Stage 4 complete:")
    for name in [cls["name"] for cls in classes_3]:
        r = s4_results[name]
        print(f"    {name:<25} {r['final_accepted']:>5} images accepted")
    print(_hr("-"))

    # ------------------------------------------------------------------
    # Checkpoint 4 -> 5
    # ------------------------------------------------------------------
    if not _confirm("Stage 4 complete. Continue to Stage 5 (Diversity Dedup + K-Selection)?"):
        print("\nStopped after Stage 4.")
        print_final_summary(classes, s2_results, s3_results, s4_results, None)
        return

    # ------------------------------------------------------------------
    # Stage 5 — Diversity dedup + K-selection
    # ------------------------------------------------------------------
    s5_results = run_stage5(classes_4, model, preprocess, tokenizer, device)

    # Warn about flagged classes but do not stop — flagged is an advisory
    flagged = [cls for cls in classes_4 if s5_results[cls["name"]]["flagged"]]
    if flagged:
        print(f"\n  {len(flagged)} class(es) flagged for not meeting their hard floor.")
        print("  Consider a scrape top-up for these classes before training.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_final_summary(classes, s2_results, s3_results, s4_results, s5_results)


if __name__ == "__main__":
    main()
