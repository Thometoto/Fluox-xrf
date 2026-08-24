#!/usr/bin/env python3
"""Interface web locale FluoX-Mo."""

import base64
import io
import os
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from flask import Flask, render_template, request  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from fluox.config import DEFAULT_LINES, SpectrumConfig  # noqa: E402
from fluox.preprocessing import read_spectrum  # noqa: E402


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
MODEL_PATHS = {
    material: PROJECT_ROOT / "models" / f"fluox-{material.lower()}.joblib"
    for material in ("Cu", "Mo", "Ag")
}


def load_model(anode_material):
    if anode_material not in MODEL_PATHS:
        raise ValueError(f"Unsupported anode material: {anode_material}")
    model_path = MODEL_PATHS[anode_material]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model is available for the {anode_material} anode."
        )
    model = joblib.load(model_path)
    if SpectrumConfig.from_dict(model.config).anode_material != anode_material:
        raise ValueError("The selected model does not match the tube anode")
    return model


def spectrum_plot(energy, counts, detected):
    figure = Figure(figsize=(12, 5.2), dpi=110, facecolor="#ffffff")
    axis = figure.add_subplot(111)
    axis.plot(energy, counts, color="#2563eb", linewidth=1.0)
    axis.set_yscale("log")
    ymax = max(float(counts.max()), 1.0)
    colors = ("#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4f46e5")
    for index, row in enumerate(detected):
        color = colors[index % len(colors)]
        for line_index, (line_energy, _) in enumerate(DEFAULT_LINES[row["element"]]):
            axis.axvline(line_energy, color=color, linestyle="--", alpha=0.62, linewidth=1)
            if line_index == 0:
                axis.text(line_energy, ymax * 0.94,
                          f"{row['element']} ({row['probability']:.2f})", rotation=90,
                          va="top", ha="right", color=color, fontsize=9)
    axis.set_xlabel("Energy (keV)")
    axis.set_ylabel("Counts (log scale)")
    axis.grid(alpha=0.16)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.route("/", methods=["GET", "POST"])
def index():
    selected_anode = request.form.get("anode", "Mo")
    context = {"predictions": None, "detected": [], "plot": None, "error": None,
               "selected_anode": selected_anode}
    if request.method == "POST":
        uploaded = request.files.get("spectrum")
        if uploaded is None or not uploaded.filename:
            context["error"] = "Please select a spectrum file."
            return render_template("index.html", **context)
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in {".csv", ".tsv", ".txt"}:
            context["error"] = "Unsupported format. Use CSV, TSV, or TXT."
            return render_template("index.html", **context)
        temporary_path = None
        try:
            model = load_model(selected_anode)
            config = SpectrumConfig.from_dict(model.config)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                uploaded.save(temporary)
                temporary_path = temporary.name
            energy, counts = read_spectrum(temporary_path, config)
            probabilities = model.predict_proba(counts)[0]
            predictions = []
            for element, probability, threshold in zip(
                model.elements, probabilities, model.thresholds
            ):
                predictions.append({
                    "element": element,
                    "probability": float(probability),
                    "percent": round(float(probability) * 100, 1),
                    "threshold": float(threshold),
                    "threshold_percent": round(float(threshold) * 100, 1),
                    "primary_energy": DEFAULT_LINES[element][0][0],
                    "line_energies": [line[0] for line in DEFAULT_LINES[element]],
                    "present": bool(probability >= threshold),
                })
            predictions.sort(key=lambda row: row["probability"], reverse=True)
            detected = [row for row in predictions if row["present"]]
            context.update(filename=uploaded.filename, predictions=predictions,
                           detected=detected, plot=spectrum_plot(energy, counts, detected))
        except Exception as exc:
            context["error"] = str(exc)
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
    return render_template("index.html", **context)


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"FluoX-Mo interface: {url}")
    if os.environ.get("FLUOX_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
