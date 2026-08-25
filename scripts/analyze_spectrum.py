#!/usr/bin/env python3
"""Analyze an XRF Energy, Counts file.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluox.config import DEFAULT_LINES, SpectrumConfig  # noqa: E402
from fluox.plotting import plot_spectrum  # noqa: E402
from fluox.preprocessing import load_spectrum, read_spectrum  # noqa: E402


def parse_args():
    examples = """examples:
  python scripts/analyze_spectrum.py data/example_mo.csv --anode Mo
  python scripts/analyze_spectrum.py data/example_mo.csv --anode Mo --show-all
  python scripts/analyze_spectrum.py spectrum.csv --anode Mo --output-dir my_results

The input file must contain two columns: Energy (keV), Counts.
The selected anode must match the anode used during acquisition.
"""
    parser = argparse.ArgumentParser(
        description="Analyze a calibrated XRF spectrum with the matching tube-anode model.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spectrum", help="Path to a CSV, TSV, or TXT Energy, Counts file")
    parser.add_argument(
        "--anode", required=True, choices=("Mo",),
        help="Tube-anode material used for the acquisition",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs"),
        help="Directory for the JSON report and annotated PNG (default: outputs/)",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print all model outputs, including elements classified as absent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spectrum_path = Path(args.spectrum).expanduser().resolve()
    model_path = PROJECT_ROOT / "models" / f"fluox-{args.anode.lower()}.joblib"
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not spectrum_path.is_file():
        print(f"Error: spectrum not found: {spectrum_path}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Error: model not found for {args.anode} anode: {model_path}", file=sys.stderr)
        return 2

    try:
        model = joblib.load(model_path)
        config = SpectrumConfig.from_dict(model.config)
        if config.anode_material != args.anode:
            raise ValueError("model and selected anode do not match")
        _, model_counts = read_spectrum(str(spectrum_path), config)
        probabilities = model.predict_proba(model_counts)[0]
        raw_energy, raw_counts = load_spectrum(str(spectrum_path))
        energy, counts = raw_energy, raw_counts
    except Exception as exc:
        print(f"Analysis error: {exc}", file=sys.stderr)
        return 1

    predictions = []
    for element, probability, threshold in zip(model.elements, probabilities, model.thresholds):
        present = bool(probability >= threshold)
        if present and element == "Ar":
            status = "atmospheric"
        elif present and element in {"Y", "Zr", "Nb"}:
            status = "uncertain"
        elif present:
            status = "present"
        else:
            status = "absent"
        predictions.append({
            "element": element,
            "energy_kev": DEFAULT_LINES[element][0][0],
            "model_score": round(float(probability), 6),
            "threshold": round(float(threshold), 6),
            "present": present,
            "status": status,
        })
    predictions.sort(key=lambda row: row["model_score"], reverse=True)
    detected = [row for row in predictions if row["present"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = spectrum_path.stem
    report_path = output_dir / f"{stem}_{args.anode.lower()}_analysis.json"
    plot_path = output_dir / f"{stem}_{args.anode.lower()}_prediction.png"
    report = {
        "spectrum": str(spectrum_path),
        "tube_anode": args.anode,
        "model": str(model_path),
        "detected_elements": [row["element"] for row in detected],
        "predictions": predictions,
        "score_definition": "Sigmoid output of a one-vs-rest logistic classifier; not an experimentally calibrated probability.",
        "threshold_definition": "Per-element threshold selected to maximize F1 on mixed synthetic and augmented-reference validation data.",
        "feature_definition": "Full 4096-channel log spectrum, estimated background, background-corrected residual, local line evidence, and Mo-region reliability weighting.",
        "training_metadata": model.config.get("training_metadata", {}),
        "warning": "The model was adapted with augmented labeled references; performance on those same source spectra is not an independent validation.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    plot_spectrum(
        energy, counts, str(plot_path), report["detected_elements"],
        title=f"XRF analysis — {spectrum_path.name} — {args.anode} anode",
        probabilities={row["element"]: row["model_score"] for row in predictions},
    )

    print(f"\nSpectrum:   {spectrum_path}")
    print(f"Tube anode: {args.anode}")
    print(f"Model:      {model_path.name}")
    print("\nDETECTED ELEMENTS")
    if detected:
        print("Element   Energy (keV)   Model score   Threshold")
        print("-------   ------------   -----------   ---------")
        for row in detected:
            print(f"{row['element']:<7}   {row['energy_kev']:>12.3f}   "
                  f"{row['model_score']:>11.3f}   {row['threshold']:>9.3f}")
    else:
        print("No element is above its decision threshold.")

    if args.show_all:
        print("\nALL MODEL OUTPUTS")
        for row in predictions:
            state = row["status"]
            print(f"{row['element']:<3}  {row['energy_kev']:>6.3f} keV  "
                  f"{row['model_score']:.3f}  {state}")

    print(f"\nPlot:   {plot_path}")
    print(f"Report: {report_path}")
    print("\nScores are not calibrated probabilities. Thresholds maximize per-element F1")
    print("on validation data and are not experimental detection limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
