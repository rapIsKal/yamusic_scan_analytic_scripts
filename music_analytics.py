"""
Yandex Music Artist Analytics
-------------------------------
Usage:
    python music_analytics.py --file your_data.jsonl

The input file should be a JSONL file (one JSON object per line),
matching the format returned by the Yandex Music artist API.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad line: {e}", file=sys.stderr)
    return records


def extract_artist(record: dict) -> dict | None:
    """Flatten one top-level record into a usable dict."""
    try:
        result = record["data"]["data"]["result"]
        artist = result["artist"]

        # skip unavailable / empty artist stubs
        if not artist.get("available", False):
            return None

        counts = artist.get("counts", {})
        stats  = result.get("stats", {})

        popular_tracks = result.get("popularTracks", [])
        albums         = result.get("albums", [])

        return {
            "id":              artist["id"],
            "name":            artist["name"],
            "genres":          artist.get("genres", []),
            "tracks":          counts.get("tracks", 0),
            "direct_albums":   counts.get("directAlbums", 0),
            "also_albums":     counts.get("alsoAlbums", 0),
            "also_tracks":     counts.get("alsoTracks", 0),
            "likes":           artist.get("likesCount", 0),
            "monthly_listeners": stats.get("lastMonthListeners", 0),
            "popular_tracks":  [
                {
                    "title":       t["title"],
                    "duration_s":  t["durationMs"] / 1000,
                    "lyrics":      t.get("lyricsAvailable", False),
                }
                for t in popular_tracks
            ],
            "albums": [
                {
                    "title": a["title"],
                    "year":  a.get("year"),
                    "tracks": a.get("trackCount", 0),
                }
                for a in albums
            ],
        }
    except (KeyError, TypeError):
        return None


# ─────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────

COLORS = plt.cm.tab10.colors


def plot_track_durations(artists: list[dict], ax: plt.Axes) -> None:
    """Bar chart: popular track durations for each artist."""
    for i, artist in enumerate(artists):
        tracks = artist["popular_tracks"]
        if not tracks:
            continue
        titles    = [t["title"] for t in tracks]
        durations = [t["duration_s"] / 60 for t in tracks]   # minutes
        x = np.arange(len(titles))
        ax.bar(x + i * 0.25, durations, width=0.25,
               label=artist["name"], color=COLORS[i % len(COLORS)], alpha=0.85)

    ax.set_title("Popular Track Durations (minutes)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Duration (min)")
    ax.set_xlabel("Track index")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_album_timeline(artists: list[dict], ax: plt.Axes) -> None:
    """Scatter plot: albums placed on a year timeline, sized by track count."""
    for i, artist in enumerate(artists):
        for album in artist["albums"]:
            year = album.get("year")
            if not year:
                continue
            ax.scatter(year, i,
                       s=album["tracks"] * 20,
                       color=COLORS[i % len(COLORS)],
                       alpha=0.75, edgecolors="white", linewidths=0.5,
                       label=artist["name"] if album == artist["albums"][0] else "")
            ax.annotate(album["title"], (year, i),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color="grey")

    ax.set_title("Album Timeline (bubble size = track count)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_yticks(range(len(artists)))
    ax.set_yticklabels([a["name"] for a in artists])
    ax.grid(axis="x", linestyle="--", alpha=0.5)


def plot_catalog_breakdown(artists: list[dict], ax: plt.Axes) -> None:
    """Grouped bar chart: tracks, direct albums, also-albums per artist."""
    names  = [a["name"] for a in artists]
    tracks = [a["tracks"]       for a in artists]
    direct = [a["direct_albums"] for a in artists]
    also   = [a["also_albums"]   for a in artists]

    x   = np.arange(len(names))
    w   = 0.25

    ax.bar(x - w,  tracks, w, label="Tracks",        color="#4C72B0", alpha=0.85)
    ax.bar(x,      direct, w, label="Direct Albums",  color="#DD8452", alpha=0.85)
    ax.bar(x + w,  also,   w, label="Also-in Albums", color="#55A868", alpha=0.85)

    ax.set_title("Catalog Breakdown per Artist", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_genre_distribution(artists: list[dict], ax: plt.Axes) -> None:
    """Pie chart of genre counts across all artists."""
    genre_counts: dict[str, int] = defaultdict(int)
    for artist in artists:
        for g in artist["genres"]:
            genre_counts[g] += 1

    if not genre_counts:
        ax.text(0.5, 0.5, "No genre data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Genre Distribution", fontsize=13, fontweight="bold")
        return

    labels = list(genre_counts.keys())
    sizes  = list(genre_counts.values())
    ax.pie(sizes, labels=labels, autopct="%1.0f%%",
           colors=COLORS[:len(labels)], startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Genre Distribution", fontsize=13, fontweight="bold")


def plot_monthly_listeners(artists: list[dict], ax: plt.Axes) -> None:
    """Horizontal bar chart of monthly listeners."""
    names     = [a["name"] for a in artists]
    listeners = [a["monthly_listeners"] for a in artists]

    y = np.arange(len(names))
    bars = ax.barh(y, listeners, color=COLORS[:len(names)], alpha=0.85)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_title("Monthly Listeners (last month)", fontsize=13, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Listeners")
    ax.grid(axis="x", linestyle="--", alpha=0.5)


# ─────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────

def build_dashboard(artists: list[dict], output_path: str = "dashboard.png") -> None:
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Yandex Music Artist Analytics Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    plot_catalog_breakdown(artists,   fig.add_subplot(gs[0, :]))   # full width
    plot_album_timeline(artists,      fig.add_subplot(gs[1, :]))   # full width
    plot_track_durations(artists,     fig.add_subplot(gs[2, 0]))
    plot_monthly_listeners(artists,   fig.add_subplot(gs[2, 1]))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Dashboard saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Yandex Music artist analytics")
    parser.add_argument("--file", required=True, help="Path to the .jsonl data file")
    parser.add_argument("--out",  default="dashboard.png", help="Output image path")
    args = parser.parse_args()

    if not Path(args.file).exists():
        sys.exit(f"[ERROR] File not found: {args.file}")

    records = load_records(args.file)
    artists = [a for r in records if (a := extract_artist(r)) is not None]

    if not artists:
        sys.exit("[ERROR] No valid artist records found in the file.")

    print(f"[INFO] Loaded {len(artists)} artist(s): {[a['name'] for a in artists]}")
    build_dashboard(artists, args.out)


if __name__ == "__main__":
    main()