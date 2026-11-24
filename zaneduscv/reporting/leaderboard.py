"""Model sonuclarini siralanmis bir tabloya cevirir."""

import pandas as pd


def make_leaderboard(results, metric):
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            f"{metric}_mean": r["mean"],
            f"{metric}_std": r["std"],
            "fit_time_s": r["fit_time"],
            "status": r.get("status", "ok"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(f"{metric}_mean", ascending=False, na_position="last").reset_index(drop=True)
