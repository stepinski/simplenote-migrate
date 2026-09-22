#!/usr/bin/env python3
"""
simplenote_migrate.py
Extract → classify → route to brain/raw or Bitwarden

Modes:
1. Automatic (default): uses LLM to classify and route notes.
2. `--out-csv <path>`: exports all notes (id, title, content) with empty TAG column.
3. `--in-csv <path>`: processes notes manually based on TAG column (B/W/D).

TAG values (can be combined):
    B  → save to Bitwarden
    W  → save to brain/raw
    D  → delete from SimpleNote after saving (or standalone for pure delete)
    (empty) → skip

Additional flags:
    --dry-run      – simulate without writing/deleting
    --limit N      – only process first N notes (for testing)
    --workers N    – parallel workers (default 4)
"""

import os
import re
import json
import csv
import base64
import subprocess
import argparse
import time
import getpass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency. Install with: pip install tqdm")
    raise

import simplenote
from openai import OpenAI

# ---------------------------
# Configuration
# ---------------------------
SN_EMAIL = os.environ["SIMPLENOTE_EMAIL"]
SN_PASSWORD = os.environ["SIMPLENOTE_PASSWORD"]
BRAIN_RAW = Path(os.environ.get("BRAIN_RAW", "~/ws/brain/raw")).expanduser()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-v4-flash-0731"
STATE_FILE = Path("migrate_state.json")
MAX_WORKERS = 4
MAX_RETRIES = 4
BASE_DELAY = 1.0

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
)

CLASSIFY_PROMPT = """Analyze this note and return JSON with:
- "type": "ordinary" or "sensitive"
- "reason": one line why
- "title": short slug-friendly title (lowercase, hyphens)
- "content": the note rewritten in clean English markdown, grammar fixed,
  language normalized. If sensitive, keep values intact but format clearly.

Sensitive means: passwords, PINs, API keys, credentials, account numbers,
secret questions, private links with tokens, or any data you'd store in a
password manager.

Note content:
{content}

Return only valid JSON, no markdown fences."""

# Global dry-run flag used by low-level helpers
DRY_RUN = False


# ---------------------------
# State management
# ---------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"done": {}}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


# ---------------------------
# Bitwarden helpers
# ---------------------------
def bw_unlock() -> str:
    password = os.environ.get("BW_PASSWORD") or getpass.getpass(
        "Bitwarden master password: "
    )
    result = subprocess.run(
        ["bw", "unlock", "--raw", password], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"bw unlock failed: {result.stderr}")
    return result.stdout.strip()


def bw_create_item(session: str, title: str, content: str):
    if DRY_RUN:
        print(f"  [dry-run] → would save to Bitwarden: {title}")
        return
    item = {"type": 2, "name": title, "notes": content, "secureNote": {"type": 0}}
    encoded = base64.b64encode(json.dumps(item).encode()).decode()
    env = {**os.environ, "BW_SESSION": session}
    result = subprocess.run(
        ["bw", "create", "item", encoded], capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"bw create failed: {result.stderr}")
    print(f"  ✓ Bitwarden: {title}")


# ---------------------------
# Retry wrapper
# ---------------------------
def call_with_retry(func, *args, retries=MAX_RETRIES, base_delay=BASE_DELAY, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2**attempt)
            print(
                f"  ↻ retry {attempt + 1}/{retries} after error: {e}; waiting {delay}s"
            )
            time.sleep(delay)


# ---------------------------
# Classification
# ---------------------------
def classify_note(content: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(content=content)}],
        temperature=0.1,
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------
# Brain saving
# ---------------------------
def save_to_brain(title: str, content: str, note_id: str):
    safe_title = re.sub(r"[^\w\-]+", "-", title).strip("-") or "note"
    filename = f"{safe_title}-{note_id[:8]}.md"
    path = BRAIN_RAW / filename
    if DRY_RUN:
        print(f"  [dry-run] → would save to brain/raw/{filename}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ brain/raw/{filename}")


# ---------------------------
# CSV helpers
# ---------------------------
def export_csv(notes_data, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["note_id", "title", "content", "TAG"])
        for note_id, title, content in notes_data:
            writer.writerow([note_id, title, content, ""])
    print(f"CSV written to {csv_path}")


def import_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("note_id"):
                rows.append(
                    {
                        "note_id": row["note_id"],
                        "title": row.get("title", ""),
                        "content": row.get("content", ""),
                        "tag": row.get("TAG", "").strip().upper(),
                    }
                )
    return rows


# ---------------------------
# Worker for automatic mode
# ---------------------------
def process_note_auto(item, bw_session, dry_run):
    note_id, mod, content = item
    if not content:
        return (note_id, mod, "skip", None, None)
    try:
        result = call_with_retry(classify_note, content)
        note_type = result["type"]
        title = result["title"]
        clean_content = result["content"]

        if note_type == "ordinary":
            save_to_brain(title, clean_content, note_id)
        else:
            call_with_retry(bw_create_item, bw_session, title, clean_content)

        return (note_id, mod, "done", title, note_type)
    except Exception as e:
        return (note_id, mod, "error", str(e), None)


# ---------------------------
# Main
# ---------------------------
def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(
        description="Migrate SimpleNote to brain/raw and Bitwarden"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write/delete anything"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only first N notes (0 = all)"
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS, help="Parallel workers"
    )
    parser.add_argument(
        "--out-csv", type=str, help="Export all notes to CSV with TAG column"
    )
    parser.add_argument(
        "--in-csv", type=str, help="Read CSV with TAG column and process manually"
    )
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    BRAIN_RAW.mkdir(parents=True, exist_ok=True)
    state = load_state()
    done_ids = state["done"]

    # Bitwarden unlock — only needed when we might write to it
    bw_session = None
    if not DRY_RUN and not args.out_csv:
        bw_session = os.environ.get("BW_SESSION") or bw_unlock()

    sn = simplenote.Simplenote(SN_EMAIL, SN_PASSWORD)

    # ---------- OUT-CSV MODE ----------
    if args.out_csv:
        notes, _ = sn.get_note_list()
        if args.limit > 0:
            notes = notes[: args.limit]

        print(f"Exporting {len(notes)} notes to CSV...")
        csv_data = []
        for meta in tqdm(notes, desc="Fetching notes"):
            note_id = meta["key"]
            note, _ = sn.get_note(note_id)
            content = note.get("content", "").strip()
            if not content:
                continue
            first_line = content.split("\n")[0]
            title = (first_line[:60] + "...") if len(first_line) > 60 else first_line
            csv_data.append((note_id, title, content))

        export_csv(csv_data, args.out_csv)
        print("Done. Fill the TAG column and re-run with --in-csv.")
        return

    # ---------- IN-CSV MODE ----------
    if args.in_csv:
        rows = import_csv(args.in_csv)
        print(f"Loaded {len(rows)} rows from {args.in_csv}")

        stats = {
            "saved_bitwarden": 0,
            "saved_brain": 0,
            "deleted": 0,
            "deleted_only": 0,  # D with no B/W
            "saved_and_deleted": 0,  # D together with B and/or W
            "skip": 0,  # empty TAG
            "error": 0,
        }

        for row in tqdm(rows, desc="Processing rows"):
            note_id = row["note_id"]
            tag = row["tag"]

            if not tag:
                stats["skip"] += 1
                continue

            saved_any = False
            deleted_now = False
            had_error = False

            # --- Bitwarden ---
            if "B" in tag:
                try:
                    call_with_retry(
                        bw_create_item, bw_session, row["title"], row["content"]
                    )
                    stats["saved_bitwarden"] += 1
                    saved_any = True
                except Exception as e:
                    stats["error"] += 1
                    had_error = True
                    print(f"  ✗ Bitwarden error for {note_id[:8]}: {e}")

            # --- Brain ---
            if "W" in tag:
                try:
                    save_to_brain(row["title"], row["content"], note_id)
                    stats["saved_brain"] += 1
                    saved_any = True
                except Exception as e:
                    stats["error"] += 1
                    had_error = True
                    print(f"  ✗ Brain error for {note_id[:8]}: {e}")

            # --- Delete (only if no error occurred for this row) ---
            if "D" in tag:
                if had_error:
                    print(f"  ⚠ skipping delete for {note_id[:8]} (had errors)")
                else:
                    try:
                        if DRY_RUN:
                            print(f"  [dry-run] → would delete {note_id[:8]}")
                        else:
                            result = sn.delete_note(note_id)
                            if result[1] != 0:
                                raise RuntimeError(result)
                            print(f"  ✓ deleted from SimpleNote: {note_id[:8]}")
                        stats["deleted"] += 1
                        deleted_now = True
                    except Exception as e:
                        stats["error"] += 1
                        print(f"  ✗ Delete error for {note_id[:8]}: {e}")

            # --- Cross-tally for clarity ---
            if deleted_now and not saved_any:
                stats["deleted_only"] += 1
            if deleted_now and saved_any:
                stats["saved_and_deleted"] += 1

        # Sanity check
        total_rows = len(rows)
        accounted = (
            stats["saved_bitwarden"]
            + stats["saved_brain"]
            + stats["skip"]
            + stats["error"]
        )
        print(f"\nDone: {json.dumps(stats, indent=2)}")
        print(f"Rows loaded: {total_rows}")
        print(
            f"Reconcile check: saved_brain + saved_bitwarden + skip + error = "
            f"{accounted} (should be ≤ {total_rows}; D rows overlap with saves)"
        )
        return

    # ---------- AUTOMATIC MODE ----------
    notes, _ = sn.get_note_list()
    print(f"Found {len(notes)} notes")

    if args.limit > 0:
        notes = notes[: args.limit]
        print(f"Processing only first {args.limit}.")

    print("Fetching note contents...")
    contents = []
    for meta in tqdm(notes, desc="Fetching notes"):
        note_id = meta["key"]
        mod = str(meta.get("modifydate", ""))

        if note_id in done_ids and done_ids[note_id] == mod:
            continue

        note, _ = sn.get_note(note_id)
        contents.append((note_id, mod, note.get("content", "").strip()))

    print(f"Notes to process: {len(contents)}")

    stats = {"ordinary": 0, "sensitive": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_note_auto, item, bw_session, DRY_RUN)
            for item in contents
        ]
        with tqdm(total=len(contents), desc="Classifying") as pbar:
            for future in as_completed(futures):
                note_id, mod, status, title, note_type = future.result()
                if status == "done":
                    state["done"][note_id] = mod
                    save_state(state)
                    stats[note_type] += 1
                elif status == "error":
                    print(f"  ✗ error on {note_id[:8]}: {title}")
                    stats["error"] += 1
                elif status == "skip":
                    stats["skipped"] += 1
                pbar.update(1)

    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
