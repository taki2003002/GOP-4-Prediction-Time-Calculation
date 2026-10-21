# =============================================================================
# Step 2: ΔT (%) Summary & Detailed Calculation Script — GOP-4
# =============================================================================
#
# Reads model prediction times from per-class CSVs under:
#   Archive/Class Prediction Time/GOP-4/prediction_times_Class*.csv
# parses VVC `encode.log` under:
#   Archive/VVC Reconstructed/All same resolution - GOP 4/
# and calculates ΔT (%) encoding time savings.
#
# GOP-4 specifics:
#   - Input  : 3 preceding frames per GOP
#   - Predict: every 4th frame (indices 4, 8, 12, 16, 20, 24, 28, 32)
#   - Per 32-frame sequence: 24 VVC-encoded frames + 8 model-predicted frames
#
# Outputs (written into the same GOP-4 folder):
#   summary_delta_t.csv, detailed_delta_t.csv
#
# Usage:
#   python delta_t_calculate_gop4.py
# =============================================================================

import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
try:
    from IPython.display import display, HTML
except ImportError:  # plain CLI / GitHub Actions
    def display(*_a, **_k):
        pass
    def HTML(x):
        return x

# Avoid Windows cp1252 crashes on 'Δ' in console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project root: THESIS_ROOT for CI, else this file's folder
_PROJECT_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parent))

# GOP-4 VVC reconstructed root (must contain encode.log). NOT the Predictions folder.
TESTING_ROOT = str(
    _PROJECT_ROOT / "Archive" / "VVC Reconstructed" / "All same resolution - GOP 4"
)

# Prediction-time CSVs from Step 1 (GOP-4 only) — per-class files from CI artifacts
PRED_DIR = str(_PROJECT_ROOT / "Archive" / "Class Prediction Time" / "GOP-4")
# Where summary / detailed ΔT CSVs are written
OUT_DIR  = PRED_DIR

# QPs and target frames configuration (GOP-4)
QPS           = ["QP37", "QP42", "QP47", "QP51"]
TARGET_FRAMES = [4, 8, 12, 16, 20, 24, 28, 32]  # last frame of each GOP-4
N_ENCODED     = 32 - len(TARGET_FRAMES)         # 24 encoded frames

CLASS_FOLDERS = ["ClassA", "ClassB", "ClassC", "ClassD"]


# =============================================================================
# PARSE ENCODE.LOG
# =============================================================================

def parse_encode_log(log_path):
    """
    Parse a VVC encode.log file and return a list of (poc, encoding_time_sec) tuples.
    The [ET <N>] field in VTM logs gives per-frame wall-clock encoding time in seconds.
    """
    ets = []
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r'\[ET\s+(\d+)\s*\]', line)
            if match:
                poc_match = re.search(r'^POC\s+(\d+)', line)
                if poc_match:
                    poc = int(poc_match.group(1))
                    et = float(match.group(1))
                    ets.append((poc, et))
    ets.sort(key=lambda x: x[0])
    return ets


def load_prediction_times(pred_dir: str) -> pd.DataFrame:
    """
    Load and concatenate prediction_times_Class*.csv (and optional merged
    prediction_times.csv) from pred_dir.
    """
    pred_path = Path(pred_dir)
    if not pred_path.is_dir():
        print(f"[WARN] Prediction dir not found: {pred_dir}")
        return pd.DataFrame()

    csv_files = sorted(pred_path.glob("prediction_times_Class*.csv"))
    merged = pred_path / "prediction_times.csv"
    if merged.is_file():
        csv_files.append(merged)

    if not csv_files:
        print(f"[WARN] No prediction_times*.csv found in {pred_dir}\n"
              f"       Place ClassA–ClassD CSVs there, then re-run.")
        return pd.DataFrame()

    frames = []
    for p in csv_files:
        df = pd.read_csv(p)
        print(f"  Loaded {len(df)} rows from {p.name}")
        frames.append(df)

    pred_df = pd.concat(frames, ignore_index=True)
    subset = [c for c in ["testing_class", "video_id", "qp", "frame_index"] if c in pred_df.columns]
    if subset:
        pred_df = pred_df.drop_duplicates(subset=subset, keep="last")
    return pred_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"[Step 2 — GOP-4] VVC root : {TESTING_ROOT}")
    print(f"[Step 2 — GOP-4] Pred dir : {PRED_DIR}\n")
    pred_df = load_prediction_times(PRED_DIR)
    if pred_df.empty:
        print("[WARN] No prediction times loaded — fallback to encode.log times for target frames.\n")
    else:
        print(f"[Step 2 — GOP-4] Using {len(pred_df)} prediction-time records.\n")

    detailed_results = []
    summary_results = []
    overall_delta_t_list = []

    # Gather video directories across all testing class folders
    video_entries = []
    for class_folder in CLASS_FOLDERS:
        class_dir = os.path.join(TESTING_ROOT, class_folder)
        if not os.path.isdir(class_dir):
            continue
        vids = sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])
        for v in vids:
            video_entries.append((class_folder, v, os.path.join(class_dir, v)))

    if not video_entries:
        print("[ERROR] No video directories found under TESTING_ROOT.")
        return

    for class_folder, video, video_dir in video_entries:
        video_delta_ts = {}

        for qp in QPS:
            log_file = os.path.join(video_dir, qp, "encode.log")
            if not os.path.exists(log_file):
                continue

            ets = parse_encode_log(log_file)
            if not ets:
                continue

            orig_time = 0.0
            enc_time = 0.0
            pred_time = 0.0

            for i, (poc, et) in enumerate(ets):
                frame_idx = i + 1  # 1-indexed frame number
                orig_time += et

                if frame_idx in TARGET_FRAMES and not pred_df.empty:
                    # Match prediction time from Step 1 (GOP-4) outcome
                    row = pred_df[(pred_df['qp'] == qp) &
                                  (pred_df['frame_index'] == frame_idx) &
                                  (pred_df['video_id'] == video)]
                    if not row.empty:
                        pred_time += float(row.iloc[0]['pred_time_sec'])
                    else:
                        pred_time += et
                elif frame_idx in TARGET_FRAMES and pred_df.empty:
                    pred_time += et
                else:
                    enc_time += et

            # prop_time = enc_time + pred_time
            prop_time = enc_time
            if orig_time > 0:
                delta_t = ((prop_time - orig_time) / orig_time) * 100
                video_delta_ts[qp] = delta_t
                overall_delta_t_list.append(delta_t)

                detailed_results.append({
                    "Class": class_folder,
                    "Video": video,
                    "QP": qp.replace("QP", ""),
                    "Original 32 (s)": f"{orig_time:.2f}",
                    f"{N_ENCODED} Encoded (s)": f"{enc_time:.2f}",
                    "Prediction (s)": f"{pred_time:.4f}",
                    "Proposed (s)": f"{prop_time:.2f}",
                    "ΔT (%)": f"{delta_t:.2f}"
                })

        if len(video_delta_ts) > 0:
            avg_dt = sum(video_delta_ts.values()) / len(video_delta_ts)
            summary_results.append({
                "Video": video,
                "ΔT @QP37": f"{video_delta_ts.get('QP37', 0):.2f}",
                "ΔT @QP42": f"{video_delta_ts.get('QP42', 0):.2f}",
                "ΔT @QP47": f"{video_delta_ts.get('QP47', 0):.2f}",
                "ΔT @QP51": f"{video_delta_ts.get('QP51', 0):.2f}",
                "Average ΔT": f"{avg_dt:.2f}"
            })

    if summary_results:
        # Add Overall Average Row
        overall_avg = sum(overall_delta_t_list) / len(overall_delta_t_list) if overall_delta_t_list else 0.0

        avg_qp37 = np.mean([float(r["ΔT @QP37"]) for r in summary_results if float(r["ΔT @QP37"]) != 0.0]) if summary_results else 0.0
        avg_qp42 = np.mean([float(r["ΔT @QP42"]) for r in summary_results if float(r["ΔT @QP42"]) != 0.0]) if summary_results else 0.0
        avg_qp47 = np.mean([float(r["ΔT @QP47"]) for r in summary_results if float(r["ΔT @QP47"]) != 0.0]) if summary_results else 0.0
        avg_qp51 = np.mean([float(r["ΔT @QP51"]) for r in summary_results if float(r["ΔT @QP51"]) != 0.0]) if summary_results else 0.0

        summary_results.append({
            "Video": "Overall Average",
            "ΔT @QP37": f"{avg_qp37:.2f}",
            "ΔT @QP42": f"{avg_qp42:.2f}",
            "ΔT @QP47": f"{avg_qp47:.2f}",
            "ΔT @QP51": f"{avg_qp51:.2f}",
            "Average ΔT": f"{overall_avg:.2f}"
        })

    df_summary = pd.DataFrame(summary_results)
    df_detailed = pd.DataFrame(detailed_results)

    # Print summary & detailed tables to console
    print("=" * 80)
    print("  SUMMARY OF ΔT (%) — GOP-4")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")

    print("=" * 80)
    print("  DETAILED BREAKDOWN — GOP-4")
    print("=" * 80)
    print(df_detailed.to_string(index=False))
    print("=" * 80 + "\n")

    # Try HTML display if running inside Jupyter/IPython environment
    try:
        display(HTML("<h3>Summary of ΔT (%) — GOP-4</h3>"))
        display(df_summary)
        display(HTML("<h3>Detailed Breakdown — GOP-4</h3>"))
        display(df_detailed)
    except Exception:
        pass

    # Save results to CSV files
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_csv = os.path.join(OUT_DIR, 'summary_delta_t.csv')
    detailed_csv = os.path.join(OUT_DIR, 'detailed_delta_t.csv')

    df_summary.to_csv(summary_csv, index=False)
    df_detailed.to_csv(detailed_csv, index=False)

    print(f"[Step 2 — GOP-4 COMPLETE] Saved results to:\n  - {summary_csv}\n  - {detailed_csv}")


if __name__ == "__main__":
    main()
