"""
Yandex Music Artist Analytics
-------------------------------
Streams JSONL into SQLite, then plots monthly listeners for all artists.

Usage:
    python music_analytics.py --file your_data.jsonl [--out listeners.png] [--db keep.db]
"""

import json
import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────

LIST_THRESHOLD = 100000

SCHEMA = """
CREATE TABLE IF NOT EXISTS artists (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    tracks            INTEGER DEFAULT 0,
    direct_albums     INTEGER DEFAULT 0,
    also_albums       INTEGER DEFAULT 0,
    also_tracks       INTEGER DEFAULT 0,
    monthly_listeners INTEGER DEFAULT 0,
    likes             INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS genres (
    artist_id TEXT,
    genre     TEXT,
    PRIMARY KEY (artist_id, genre)
);

CREATE TABLE IF NOT EXISTS albums (
    artist_id   TEXT,
    title       TEXT,
    year        INTEGER,
    track_count INTEGER DEFAULT 0,
    PRIMARY KEY (artist_id, title)
);

CREATE TABLE IF NOT EXISTS popular_tracks (
    artist_id  TEXT,
    position   INTEGER,
    title      TEXT,
    duration_s REAL,
    PRIMARY KEY (artist_id, position)
);
"""

def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


# ─────────────────────────────────────────────
# Streaming ingest
# ─────────────────────────────────────────────

def iter_records(path: str):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno} skipped: {e}", file=sys.stderr)


def ingest(path: str, con: sqlite3.Connection) -> tuple[int, int, int]:
    total = skipped = more_threshold = 0

    for record in iter_records(path):
        total += 1
        try:
            result = record["data"]["data"]["result"]
            artist = result["artist"]
            if not artist.get("available", False):
                skipped += 1
                continue

            aid    = artist["id"]
            name   = artist["name"]
            counts = artist.get("counts", {})
            stats  = result.get("stats", {})

            con.execute("""
                INSERT INTO artists
                    (id, name, tracks, direct_albums, also_albums, also_tracks,
                     monthly_listeners, likes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    tracks            = MAX(tracks,            excluded.tracks),
                    direct_albums     = MAX(direct_albums,     excluded.direct_albums),
                    also_albums       = MAX(also_albums,       excluded.also_albums),
                    also_tracks       = MAX(also_tracks,       excluded.also_tracks),
                    monthly_listeners = MAX(monthly_listeners, excluded.monthly_listeners),
                    likes             = MAX(likes,             excluded.likes)
            """, (
                aid, name,
                counts.get("tracks", 0),
                counts.get("directAlbums", 0),
                counts.get("alsoAlbums", 0),
                counts.get("alsoTracks", 0),
                stats.get("lastMonthListeners", 0),
                artist.get("likesCount", 0),
            ))

            for g in artist.get("genres", []):
                con.execute("INSERT OR IGNORE INTO genres VALUES (?, ?)", (aid, g))

            for a in result.get("albums", []):
                con.execute("""
                    INSERT OR IGNORE INTO albums (artist_id, title, year, track_count)
                    VALUES (?, ?, ?, ?)
                """, (aid, a.get("title", ""), a.get("year"), a.get("trackCount", 0)))

            existing = con.execute(
                "SELECT COUNT(*) FROM popular_tracks WHERE artist_id=?", (aid,)
            ).fetchone()[0]
            if existing == 0:
                for pos, t in enumerate(result.get("popularTracks", [])):
                    con.execute("""
                        INSERT OR IGNORE INTO popular_tracks
                            (artist_id, position, title, duration_s)
                        VALUES (?, ?, ?, ?)
                    """, (aid, pos, t.get("title", ""), t.get("durationMs", 0) / 1000))
            if stats.get("lastMonthListeners", 0) >= LIST_THRESHOLD:
                more_threshold += 1
        except (KeyError, TypeError):
            skipped += 1
            continue

        if total % 5000 == 0:
            con.commit()
            print(f"  … {total} records processed", flush=True)

    con.commit()
    return total, skipped, more_threshold


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot_monthly_listeners(con: sqlite3.Connection, output_path: str) -> None:
    """
    Histogram of monthly listeners across ALL artists.
    Cursor-iterated — one value at a time, never a full list in memory.
    """
    listeners = []
    for (val,) in con.execute(
        "SELECT monthly_listeners FROM artists WHERE monthly_listeners > 0"
    ):
        listeners.append(val)

    if not listeners:
        sys.exit("[ERROR] No listener data found.")

    fig, axes = plt.subplots(1, 1, figsize=(14, 5))
    fig.suptitle(f"Monthly Listeners — {len(listeners):,} artists",
                 fontsize=14, fontweight="bold")

    # Log scale — more useful when a few artists dominate
    bin_width = 20000
    bins = np.arange(0, max(listeners) + bin_width, bin_width)
    axes.hist(listeners, bins=bins, color="#DD8452", alpha=0.85, edgecolor="white", log=True)
    axes.set_title("Log scale  (shows long tail)")
    axes.set_xlabel("Listeners last month")
    axes.set_ylabel("Number of artists (log)")
    axes.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Chart saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Yandex Music — monthly listeners chart")
    parser.add_argument("--file", required=True, help="Path to the .jsonl data file")
    parser.add_argument("--out",  default="listeners.png", help="Output image path")
    parser.add_argument("--db",   default=None,
                        help="SQLite DB path (default: temp file, deleted after run)")
    args = parser.parse_args()

    if not Path(args.file).exists():
        sys.exit(f"[ERROR] File not found: {args.file}")

    use_temp = args.db is None
    db_path  = args.db or tempfile.mktemp(suffix=".db")

    try:
        con = open_db(db_path)
        print(f"[INFO] Ingesting {args.file} …")
        total, skipped, more_threshold = ingest(args.file, con)
        print(f"[INFO] Done — {total} records, {skipped} skipped, {more_threshold} is more threshold {LIST_THRESHOLD}")
        plot_monthly_listeners(con, args.out)
        con.close()
    finally:
        if use_temp:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()