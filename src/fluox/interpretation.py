from typing import Dict, Iterable, List

import numpy as np

from .baseline import multi_line_evidence
from .config import DEFAULT_LINES, SpectrumConfig


def _line_types(element: str, line_count: int) -> List[str]:
    if element == "Pb":
        labels = ["Lα", "Lβ", "Lγ"]
    elif element == "Ba":
        labels = ["Lα", "Lβ"]
    else:
        labels = ["Kα", "Kβ"]
    return labels[:line_count]


def interpret_predictions(model, config: SpectrumConfig, counts: np.ndarray) -> List[dict]:
    """Combine classifier scores with interpretable multi-line peak evidence."""
    probabilities = model.predict_proba(counts)[0]
    evidence = multi_line_evidence(counts, config)
    rows = []
    for element, probability, threshold in zip(
        model.elements, probabilities, model.thresholds
    ):
        peak = evidence[element]
        ratio = float(probability) / max(float(threshold), 1e-6)
        model_present = bool(probability >= threshold)
        # Peak evidence qualifies a classifier decision but never creates one
        # by itself: crowded XRF spectra contain many overlapping line windows.
        present = model_present
        trace = not present and ratio >= 0.70 and peak["max_snr"] >= 3.0
        if present:
            status = "present"
        elif trace:
            status = "trace"
        else:
            status = "absent"
        lines = DEFAULT_LINES[element]
        line_types = _line_types(element, len(lines))
        primary_index = int(np.argmax([intensity for _, intensity in lines]))
        rows.append({
            "element": element,
            "probability": float(probability),
            "percent": round(float(probability) * 100, 1),
            "threshold": float(threshold),
            "threshold_percent": round(float(threshold) * 100, 1),
            "primary_energy": lines[primary_index][0],
            "primary_line_type": line_types[primary_index],
            "line_energies": [line[0] for line in lines],
            "line_types": line_types,
            "line_snrs": [round(value, 3) for value in peak["line_snrs"]],
            "supported_lines": peak["supported_lines"],
            "peak_snr": round(peak["max_snr"], 3),
            "present": present,
            "active": present or trace,
            "status": status,
        })

    by_element: Dict[str, dict] = {row["element"]: row for row in rows}
    if by_element.get("Ar", {}).get("active"):
        by_element["Ar"]["status"] = "atmospheric"
    for element in ("Y", "Zr", "Nb"):
        if by_element.get(element, {}).get("active"):
            by_element[element]["status"] = "uncertain"
    if by_element.get("As", {}).get("active") and any(
        by_element.get(element, {}).get("active") for element in ("Pb", "Br")
    ):
        by_element["As"]["status"] = "ambiguous"
    if by_element.get("Ba", {}).get("active") and by_element.get("Ti", {}).get("active"):
        by_element["Ba"]["status"] = "ambiguous"

    for row in rows:
        if row["status"] == "present":
            ratio = row["probability"] / max(row["threshold"], 1e-6)
            row["evidence_label"] = "strong evidence" if (
                ratio >= 2.0 or row["supported_lines"] >= 2
            ) else "detected"
        elif row["status"] == "atmospheric":
            row["evidence_label"] = "air signal"
        elif row["status"] == "trace":
            row["evidence_label"] = "possible trace"
        elif row["status"] == "ambiguous":
            row["evidence_label"] = "ambiguous"
        elif row["status"] == "uncertain":
            row["evidence_label"] = "uncertain"
        else:
            row["evidence_label"] = "not detected"

    rows.sort(key=lambda row: (not row["active"], -row["probability"]))
    return rows


def active_elements(rows: Iterable[dict]) -> List[str]:
    return [row["element"] for row in rows if row["active"]]
