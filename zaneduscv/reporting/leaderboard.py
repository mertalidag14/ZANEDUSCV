"""Model sonuclarini siralanmis bir tabloya cevirir."""

import pandas as pd


def make_leaderboard(results, metric, greater_is_better=True):
    rows = [{
        "model": name,
        f"{metric}_mean": r["mean"],
        f"{metric}_std": r["std"],
        "fit_time_s": r["fit_time"],
        "status": r.get("status", "ok"),
    } for name, r in results.items()]
    df = pd.DataFrame(rows)
    return df.sort_values(f"{metric}_mean", ascending=not greater_is_better,
                          na_position="last").reset_index(drop=True)
