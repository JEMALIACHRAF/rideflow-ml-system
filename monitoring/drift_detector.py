"""
Drift monitoring with Evidently.
Detects: data drift (feature distribution shift), concept drift (target shift),
and model performance degradation (PSI, MAPE regression).
Sends alerts when thresholds are exceeded.
"""
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    from evidently.metrics import (
        DatasetDriftMetric, ColumnDriftMetric,
        RegressionQualityMetric,
    )
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False
    logger.warning("Evidently not installed — using fallback PSI monitoring")


# ─── PSI fallback (no external deps) ─────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    ref_pct = np.histogram(reference, bins=bins)[0] / len(reference)
    cur_pct = np.histogram(current,   bins=bins)[0] / len(current)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ─── Main DriftMonitor ────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Monitors feature drift and model performance in production.

    Usage:
        monitor = DriftMonitor(reference_df, output_dir="reports/monitoring")
        result  = monitor.run(current_df, y_true, y_pred)
        if result["alert"]:
            monitor.send_alert(result)
    """
    def __init__(self, reference_df: pd.DataFrame,
                 output_dir: str = "reports/monitoring",
                 drift_threshold: float = 0.2,
                 mape_threshold: float = 0.20,
                 psi_threshold:  float = 0.2):
        self.reference_df    = reference_df
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold
        self.mape_threshold  = mape_threshold
        self.psi_threshold   = psi_threshold
        self.history: list[dict] = []

    def check_feature_drift(self, current_df: pd.DataFrame) -> dict:
        """Compute PSI for every numeric feature. Return dict of results."""
        results = {}
        for col in self.reference_df.select_dtypes(include=[np.number]).columns:
            if col not in current_df.columns:
                continue
            ref = self.reference_df[col].dropna().values
            cur = current_df[col].dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue
            psi_val = _psi(ref, cur)
            status  = "stable" if psi_val < 0.1 else "warn" if psi_val < self.psi_threshold else "drift"
            results[col] = {"psi": round(psi_val, 4), "status": status}
        return results

    def check_target_drift(self, y_ref: np.ndarray, y_cur: np.ndarray) -> dict:
        """PSI on target distribution."""
        psi_val = _psi(y_ref, y_cur)
        ks_stat = float(np.max(np.abs(
            np.sort(y_ref).cumsum() / y_ref.sum() -
            np.sort(y_cur).cumsum() / y_cur.sum()
        ))) if len(y_ref) > 1 and len(y_cur) > 1 else 0.0

        return {
            "psi":    round(psi_val, 4),
            "ks":     round(ks_stat, 4),
            "status": "stable" if psi_val < 0.1 else "warn" if psi_val < 0.2 else "drift",
        }

    def check_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute current MAPE and RMSE."""
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1.0))))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {
            "mape":   round(mape, 4),
            "rmse":   round(rmse, 4),
            "alert":  mape > self.mape_threshold,
        }

    def run(self, current_df: pd.DataFrame,
            y_true: np.ndarray | None = None,
            y_pred: np.ndarray | None = None) -> dict:
        """Full monitoring run. Returns summary dict with alert flag."""
        timestamp = datetime.now().isoformat()
        feature_drift = self.check_feature_drift(current_df)

        drifted_features = [f for f, r in feature_drift.items() if r["status"] == "drift"]
        warned_features  = [f for f, r in feature_drift.items() if r["status"] == "warn"]

        perf = {}
        if y_true is not None and y_pred is not None:
            perf = self.check_performance(y_true, y_pred)

        alert = len(drifted_features) > 0 or perf.get("alert", False)

        result = {
            "timestamp":        timestamp,
            "n_features_checked": len(feature_drift),
            "drifted_features": drifted_features,
            "warned_features":  warned_features,
            "feature_drift":    feature_drift,
            "performance":      perf,
            "alert":            alert,
            "recommendation":   (
                "RETRAIN MODEL — significant drift detected" if alert else
                "MONITOR — slight drift in some features" if warned_features else
                "OK — distributions stable"
            ),
        }

        self.history.append(result)
        self._save_result(result)

        level = "ERROR" if alert else "WARNING" if warned_features else "SUCCESS"
        getattr(logger, level.lower())(
            f"Drift check: {len(drifted_features)} drifted, "
            f"{len(warned_features)} warned — {result['recommendation']}"
        )
        return result

    def _save_result(self, result: dict) -> None:
        """Append result to JSONL log."""
        log_path = self.output_dir / "drift_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def run_evidently_report(self, current_df: pd.DataFrame,
                              y_true: pd.Series | None = None,
                              y_pred: pd.Series | None = None) -> Path:
        """Generate full Evidently HTML report (requires evidently installed)."""
        if not EVIDENTLY_OK:
            logger.warning("Evidently not available — skipping full report")
            return Path()

        ref = self.reference_df.copy()
        cur = current_df.copy()
        if y_true is not None:
            cur["target"] = y_true.values
            ref["target"] = ref["demand"].values if "demand" in ref.columns else 0
        if y_pred is not None:
            cur["prediction"] = y_pred.values

        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=ref, current_data=cur)

        out = self.output_dir / f"evidently_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        report.save_html(str(out))
        logger.success(f"Evidently report saved → {out}")
        return out

    def get_drift_trend(self, feature: str, last_n: int = 10) -> pd.DataFrame:
        """Extract PSI trend for a specific feature from history."""
        rows = []
        for entry in self.history[-last_n:]:
            psi_val = entry.get("feature_drift", {}).get(feature, {}).get("psi", None)
            if psi_val is not None:
                rows.append({"timestamp": entry["timestamp"], "psi": psi_val})
        return pd.DataFrame(rows)

    def send_alert(self, result: dict, smtp_host: str = "localhost",
                   from_addr: str = "rideflow@ml.com",
                   to_addr: str = "mlops@company.com") -> None:
        """Send email alert when drift or performance degradation is detected."""
        body = (
            f"🚨 RideFlow ML Alert — {result['timestamp']}\n\n"
            f"Recommendation: {result['recommendation']}\n\n"
            f"Drifted features ({len(result['drifted_features'])}): "
            f"{', '.join(result['drifted_features'])}\n\n"
            f"Performance: {result.get('performance', {})}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"[RideFlow] ML Drift Alert — {result['recommendation']}"
        msg["From"] = from_addr
        msg["To"]   = to_addr
        try:
            with smtplib.SMTP(smtp_host) as s:
                s.send_message(msg)
            logger.info(f"Alert sent to {to_addr}")
        except Exception as e:
            logger.warning(f"Alert email failed: {e}")
