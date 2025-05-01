from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["visualize_session"]


def _ensure_outdir(outdir: os.PathLike | str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def visualize_session(
    hand_winnings: Sequence[int | float],
    stats: Mapping[str, float | int],
    outdir: os.PathLike | str = "plots",
    dpi: int = 150,
) -> list[Path]:
    """Create and save a set of standard session graphics.

    Parameters
    ----------
    hand_winnings
        Per‑hand profit/loss values (chips).  *Positive = we won the hand.*
    stats
        The summary dict returned by ``SlumbotIntegration.play_session``.  Only
        ``hands_played`` and ``win_rate`` are used for labels, but you can add
        more.
    outdir
        Destination folder for the PNGs.  Created if it doesn’t exist.
    dpi
        Target resolution.

    Returns
    -------
    list[pathlib.Path]
        A list with the three filenames in creation order.
    """

    if not hand_winnings:
        raise ValueError("hand_winnings is empty – nothing to plot ⛔")

    out = _ensure_outdir(outdir)

    # DataFrame makes cumulative sums & filtering trivial
    df = pd.DataFrame({
        "hand": range(1, len(hand_winnings) + 1),
        "winnings": list(hand_winnings),
    })
    df["cumulative"] = df["winnings"].cumsum()

    paths: list[Path] = []

    # 1️⃣  cumulative winnings line chart
    plt.figure()
    plt.plot(df["hand"], df["cumulative"], linewidth=1.4)
    plt.xlabel("Hand #")
    plt.ylabel("Cumulative winnings (chips)")
    plt.title("Cumulative winnings over session")
    paths.append(out / "cumulative_winnings.png")
    plt.savefig(paths[-1], dpi=dpi, bbox_inches="tight")
    plt.close()

    # 2️⃣  per‑hand bar chart
    plt.figure()
    plt.bar(df["hand"], df["winnings"], width=0.9)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Hand #")
    plt.ylabel("Winnings for hand (chips)")
    plt.title("Profit / loss on each hand")
    paths.append(out / "per_hand_winnings.png")
    plt.savefig(paths[-1], dpi=dpi, bbox_inches="tight")
    plt.close()

    # 3️⃣  histogram of hand results
    plt.figure()
    plt.hist(df["winnings"], bins="auto")
    plt.xlabel("Hand result (chips)")
    plt.ylabel("Frequency")
    plt.title("Distribution of individual‑hand outcomes")
    paths.append(out / "winnings_histogram.png")
    plt.savefig(paths[-1], dpi=dpi, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    for p in paths:
        print("  ", p)

    return paths