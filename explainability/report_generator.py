"""
Explainability report generator.
Produces a standalone HTML report combining SHAP, LIME, PDP, Anchors, DiCE.
No external dependencies needed to open — all assets embedded as base64.
"""
import base64
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger


def _img_to_b64(path: Path) -> str:
    """Encode a PNG image to base64 data URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def generate_report(
    model_name: str,
    shap_importance: pd.Series,
    metrics: dict,
    shap_dir: Path,
    lime_dir: Path,
    pdp_dir: Path,
    anchor_result: dict | None = None,
    cf_result: pd.DataFrame | None = None,
    output_path: Path = Path("reports/explainability_report.html"),
) -> Path:
    """
    Generate a full standalone HTML explainability report.

    Args:
        model_name:       Name of the model being explained.
        shap_importance:  pd.Series of mean |SHAP| per feature.
        metrics:          Dict of evaluation metrics (rmse, mape, etc.).
        shap_dir:         Directory containing SHAP PNG outputs.
        lime_dir:         Directory containing LIME PNG outputs.
        pdp_dir:          Directory containing PDP/ICE PNG outputs.
        anchor_result:    Dict from AnchorExplainer.explain().
        cf_result:        DataFrame from CounterfactualExplainer.generate().
        output_path:      Where to save the HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load images
    imgs = {
        "shap_summary":   _img_to_b64(shap_dir / "shap_summary.png"),
        "shap_bar":       _img_to_b64(shap_dir / "shap_bar_importance.png"),
        "shap_waterfall": _img_to_b64(shap_dir / "shap_waterfall.png"),
        "lime_0":         _img_to_b64(lime_dir / "lime_explanation.png"),
        "pdp_hour":       _img_to_b64(pdp_dir / "pdp_hour.png"),
        "ice_hour":       _img_to_b64(pdp_dir / "ice_hour.png"),
    }

    # Build top-10 feature table
    top_features_html = "".join(
        f"<tr><td>{i+1}</td><td><code>{feat}</code></td>"
        f"<td>{val:.4f}</td></tr>"
        for i, (feat, val) in enumerate(shap_importance.head(10).items())
    )

    # Metrics table
    metrics_html = "".join(
        f"<tr><td><b>{k.upper()}</b></td><td>{v}</td></tr>"
        for k, v in metrics.items()
    )

    # Anchor section
    anchor_html = ""
    if anchor_result:
        anchor_html = f"""
        <div class="card green">
          <h3>⚓ Anchor Rule</h3>
          <code class="rule">{anchor_result.get('rule_str', 'N/A')}</code>
          <div class="meta">
            Precision: <b>{anchor_result.get('precision', 0):.1%}</b> &nbsp;|&nbsp;
            Coverage: <b>{anchor_result.get('coverage', 0):.1%}</b>
          </div>
          <p>Interpretation: when these conditions hold, the model's prediction
          is stable with {anchor_result.get('precision', 0):.0%} probability,
          regardless of other feature values.</p>
        </div>"""

    # Counterfactual section
    cf_html = ""
    if cf_result is not None and len(cf_result) > 0:
        cf_rows = ""
        for _, row in cf_result.iterrows():
            changes_str = "<br>".join(f"<code>{k}: {v}</code>"
                                       for k, v in row.get("changes", {}).items())
            cf_rows += (f"<tr><td>{row['cf_prediction']}</td>"
                        f"<td>{row['n_changes']}</td>"
                        f"<td>{changes_str}</td></tr>")
        cf_html = f"""
        <div class="card">
          <h3>🔀 Counterfactual Explanations (DiCE)</h3>
          <p>Minimal feature changes that would achieve the desired demand range:</p>
          <table><thead><tr>
            <th>CF Prediction</th><th># Changes</th><th>What would need to change</th>
          </tr></thead><tbody>{cf_rows}</tbody></table>
        </div>"""

    def img_section(title: str, key: str, caption: str = "") -> str:
        src = imgs.get(key, "")
        if not src:
            return f"<p><i>Image not found: {key}</i></p>"
        return f"""
        <div class="img-block">
          <h4>{title}</h4>
          <img src="{src}" alt="{title}">
          {f'<p class="caption">{caption}</p>' if caption else ''}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RideFlow Explainability Report — {model_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8fafc; color: #1e293b; line-height: 1.6; }}
  header {{ background: linear-gradient(135deg, #1e40af, #7c3aed);
            color: white; padding: 2rem 3rem; }}
  header h1 {{ font-size: 1.8rem; font-weight: 700; }}
  header p  {{ opacity: 0.85; margin-top: 0.3rem; }}
  .container {{ max-width: 1100px; margin: 2rem auto; padding: 0 2rem; }}
  .section {{ margin-bottom: 2.5rem; }}
  .section h2 {{ font-size: 1.3rem; font-weight: 700; color: #1e40af;
                 border-bottom: 3px solid #bfdbfe; padding-bottom: 0.4rem;
                 margin-bottom: 1rem; }}
  .card {{ background: white; border-radius: 10px; padding: 1.5rem;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1rem; }}
  .card.green {{ border-left: 4px solid #16a34a; }}
  .card h3 {{ font-size: 1rem; margin-bottom: 0.8rem; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                   gap: 1rem; }}
  .metric {{ background: white; border-radius: 10px; padding: 1rem 1.2rem;
             text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .metric .val {{ font-size: 1.6rem; font-weight: 700; color: #1e40af; }}
  .metric .lbl {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th, td {{ padding: 0.5rem 0.8rem; text-align: left;
            border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  img {{ max-width: 100%; border-radius: 8px; margin-top: 0.5rem; }}
  .img-block {{ background: white; border-radius: 10px; padding: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1rem; }}
  .img-block h4 {{ margin-bottom: 0.5rem; color: #374151; }}
  .caption {{ font-size: 0.8rem; color: #64748b; margin-top: 0.4rem; }}
  code {{ background: #f1f5f9; padding: 0.1rem 0.4rem; border-radius: 4px;
          font-size: 0.85rem; }}
  .rule {{ display: block; background: #f0fdf4; border: 1px solid #86efac;
           padding: 0.8rem 1rem; border-radius: 6px; font-size: 0.9rem;
           margin: 0.5rem 0; }}
  .meta {{ color: #374151; margin-top: 0.5rem; font-size: 0.88rem; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  footer {{ text-align: center; padding: 2rem; color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
<header>
  <h1>🚗 RideFlow ML — Explainability Report</h1>
  <p>Model: <b>{model_name}</b> &nbsp;|&nbsp;
     Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
     RideFlow ML System</p>
</header>

<div class="container">

  <!-- 1. Model Performance -->
  <div class="section">
    <h2>1. Model Performance</h2>
    <div class="metrics-grid">
      {"".join(f'<div class="metric"><div class="val">{v}</div><div class="lbl">{k.upper()}</div></div>' for k, v in metrics.items())}
    </div>
  </div>

  <!-- 2. SHAP Global Importance -->
  <div class="section">
    <h2>2. Global Feature Importance (SHAP)</h2>
    <div class="card">
      <h3>Top 10 Features by Mean |SHAP|</h3>
      <table><thead><tr><th>#</th><th>Feature</th><th>Mean |SHAP|</th></tr></thead>
      <tbody>{top_features_html}</tbody></table>
    </div>
    <div class="two-col">
      {img_section("Beeswarm Summary", "shap_summary",
                   "Each dot = one sample. X-axis = SHAP value. Colour = feature value.")}
      {img_section("Bar Importance", "shap_bar",
                   "Mean absolute SHAP value across all samples.")}
    </div>
  </div>

  <!-- 3. Local SHAP Explanation -->
  <div class="section">
    <h2>3. Local Explanation — Single Prediction (SHAP Waterfall)</h2>
    {img_section("Waterfall Plot", "shap_waterfall",
                 "Starting from the base value (average prediction), each bar shows a feature's contribution to this specific prediction.")}
  </div>

  <!-- 4. LIME -->
  <div class="section">
    <h2>4. Local Explanation — LIME</h2>
    {img_section("LIME Local Model", "lime_0",
                 "A local linear model trained around this prediction. Positive bars increase demand, negative bars decrease it.")}
  </div>

  <!-- 5. PDP / ICE -->
  <div class="section">
    <h2>5. Marginal Effects (PDP & ICE)</h2>
    <div class="two-col">
      {img_section("PDP — Hour of Day", "pdp_hour",
                   "Average predicted demand across all zones as a function of hour.")}
      {img_section("ICE — Hour of Day", "ice_hour",
                   "Individual curves (grey) show per-zone effect. Red = population average.")}
    </div>
  </div>

  <!-- 6. Anchors -->
  <div class="section">
    <h2>6. Anchor (IF-THEN Rule)</h2>
    {anchor_html or '<div class="card"><p>No anchor result provided.</p></div>'}
  </div>

  <!-- 7. Counterfactuals -->
  <div class="section">
    <h2>7. Counterfactual Explanations (DiCE)</h2>
    {cf_html or '<div class="card"><p>No counterfactual result provided.</p></div>'}
  </div>

</div>
<footer>RideFlow ML System — Explainability Report — {datetime.now().year}</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.success(f"Explainability report saved → {output_path}")
    return output_path
