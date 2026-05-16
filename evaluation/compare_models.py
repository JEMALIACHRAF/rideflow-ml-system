"""
compare_models.py — Standalone benchmark script.
Trains all models, evaluates, and produces a rich HTML comparison report.

Usage: python evaluation/compare_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from evaluation.metrics import compute_all_metrics


def build_comparison_report(results: dict[str, dict],
                             output_path: Path = Path("reports/model_comparison.html")) -> Path:
    """
    Generate a clean HTML benchmark report comparing all models.

    Args:
        results: {model_name: {metric_name: value, ...}}
        output_path: where to write the HTML.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by MAPE ascending
    sorted_models = sorted(results.items(), key=lambda x: x[1].get("mape", 999))
    best_name     = sorted_models[0][0]

    def metric_cell(val: float, best: float, metric: str) -> str:
        """Colour-code a metric cell: green if best, yellow if within 5%, red otherwise."""
        is_lower_better = metric in {"mape", "rmse", "mae"}
        if (is_lower_better and val == best) or (not is_lower_better and val == best):
            cls = "best"
        elif is_lower_better and val <= best * 1.05:
            cls = "close"
        else:
            cls = ""
        fmt = f"{val:.3%}" if metric == "mape" else f"{val:.4f}"
        return f'<td class="{cls}">{fmt}</td>'

    metric_names = ["mape", "rmse", "mae", "r2", "pinball_90"]
    best_vals    = {m: min(r.get(m, 999) for _, r in sorted_models)
                    if m != "r2" else max(r.get(m, -999) for _, r in sorted_models)
                    for m in metric_names}

    rows_html = ""
    for i, (name, m) in enumerate(sorted_models):
        rank_badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        rows_html += f"<tr><td>{rank_badge} <b>{name}</b></td>"
        for metric in metric_names:
            val = m.get(metric, float("nan"))
            rows_html += metric_cell(val, best_vals[metric], metric)
        rows_html += "</tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RideFlow — Model Comparison Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8fafc; color: #1e293b; }}
  header {{ background: linear-gradient(135deg,#1e40af,#7c3aed); color:white;
            padding: 2rem 3rem; }}
  header h1 {{ font-size: 1.7rem; font-weight: 700; }}
  header p  {{ opacity:.85; margin-top:.3rem; }}
  .container {{ max-width: 960px; margin: 2rem auto; padding: 0 2rem; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th {{ background: #1e40af; color: white; padding: .7rem 1rem;
        text-align: left; font-size:.88rem; }}
  td {{ padding: .65rem 1rem; border-bottom: 1px solid #e2e8f0; font-size:.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  td.best  {{ background: #dcfce7; color: #166534; font-weight: 700; }}
  td.close {{ background: #fef9c3; color: #854d0e; }}
  .legend {{ display: flex; gap: 1rem; margin-top: .8rem; font-size:.8rem; }}
  .badge {{ padding:.2rem .6rem; border-radius:4px; }}
  .b-best  {{ background:#dcfce7; color:#166534; }}
  .b-close {{ background:#fef9c3; color:#854d0e; }}
  footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem; }}
</style>
</head>
<body>
<header>
  <h1>🚗 RideFlow — Model Benchmark Report</h1>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Best model: {best_name}</p>
</header>
<div class="container">
  <h2 style="margin:1.5rem 0 .8rem;font-size:1.1rem;color:#1e40af;">
    Test Set Performance — All Models
  </h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>MAPE ↓</th><th>RMSE ↓</th><th>MAE ↓</th>
        <th>R² ↑</th><th>Pinball@90 ↓</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="legend">
    <span class="badge b-best">🟢 Best in column</span>
    <span class="badge b-close">🟡 Within 5% of best</span>
    <span>↓ Lower is better &nbsp; ↑ Higher is better</span>
  </div>

  <h2 style="margin:2rem 0 .8rem;font-size:1.1rem;color:#1e40af;">
    Metric Definitions
  </h2>
  <table>
    <thead><tr><th>Metric</th><th>Formula</th><th>Interpretation</th></tr></thead>
    <tbody>
      <tr><td><b>MAPE</b></td>
          <td>mean |y - ŷ| / (|y| + 1)</td>
          <td>% error relative to actual demand</td></tr>
      <tr><td><b>RMSE</b></td>
          <td>√mean(y - ŷ)²</td>
          <td>Penalises large errors heavily</td></tr>
      <tr><td><b>MAE</b></td>
          <td>mean |y - ŷ|</td>
          <td>Average absolute error in rides/hour</td></tr>
      <tr><td><b>R²</b></td>
          <td>1 - SS_res/SS_tot</td>
          <td>Variance explained (1.0 = perfect)</td></tr>
      <tr><td><b>Pinball@90</b></td>
          <td>asymmetric quantile loss (q=0.9)</td>
          <td>Penalises under-prediction 9× more (demand safety buffer)</td></tr>
    </tbody>
  </table>
</div>
<footer>RideFlow ML System — {datetime.now().year}</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.success(f"Comparison report saved → {output_path}")
    return output_path


if __name__ == "__main__":
    # Demo with mock metrics (replace with actual trained model outputs)
    mock_results = {
        "LightGBM":       {"mape": 0.092, "rmse": 3.41, "mae": 2.18, "r2": 0.891, "pinball_90": 1.87},
        "XGBoost":        {"mape": 0.098, "rmse": 3.67, "mae": 2.31, "r2": 0.874, "pinball_90": 2.01},
        "CatBoost":       {"mape": 0.094, "rmse": 3.52, "mae": 2.24, "r2": 0.884, "pinball_90": 1.93},
        "VotingEnsemble": {"mape": 0.089, "rmse": 3.28, "mae": 2.09, "r2": 0.899, "pinball_90": 1.79},
        "WeightedBlend":  {"mape": 0.087, "rmse": 3.19, "mae": 2.03, "r2": 0.905, "pinball_90": 1.74},
        "Stacking":       {"mape": 0.085, "rmse": 3.11, "mae": 1.97, "r2": 0.911, "pinball_90": 1.71},
    }
    build_comparison_report(mock_results, Path("reports/model_comparison.html"))
