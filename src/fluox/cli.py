import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from sklearn.model_selection import train_test_split

from .baseline import peak_scores
from .config import SpectrumConfig
from .model import fit_model, metrics
from .plotting import plot_spectrum
from .preprocessing import load_spectrum, read_spectrum
from .simulator import energy_grid, generate_dataset, simulate_spectrum


def _background_templates(paths, config):
    if not paths:
        return None
    grid = np.linspace(config.energy_min_kev, config.energy_max_kev, config.channels)
    templates = []
    for path in paths:
        energy, counts = load_spectrum(path)
        sampled = np.interp(grid, energy, counts)
        # Remove narrow fluorescence peaks while retaining the measured
        # continuum and the broad rise caused by Mo scattering.
        points_per_kev = config.channels / (config.energy_max_kev - config.energy_min_kev)
        window = max(31, int(round(0.55 * points_per_kev)) | 1)
        baseline = percentile_filter(sampled, percentile=25, size=window, mode="nearest")
        baseline = gaussian_filter1d(baseline, sigma=max(2.0, points_per_kev * 0.035))
        templates.append(np.maximum(baseline, 0.0))
    return np.asarray(templates, dtype=np.float32)


def _parse_labeled_references(specifications, config):
    references = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("A labeled reference must use PATH=Si,P,Fe format")
        path, elements_text = specification.split("=", 1)
        elements = [item.strip() for item in elements_text.split(",") if item.strip()]
        unknown = sorted(set(elements) - set(config.elements))
        if unknown:
            raise ValueError(f"Unsupported labeled elements: {', '.join(unknown)}")
        _, counts = read_spectrum(path, config)
        labels = np.zeros(len(config.elements), dtype=np.uint8)
        for element in elements:
            labels[config.elements.index(element)] = 1
        references.append((counts.astype(np.float64), labels))
    return references


def _add_real_augmentations(x, y, references, fraction, seed, config):
    if not references or fraction <= 0:
        return x, y
    rng = np.random.default_rng(seed + 7919)
    count = min(len(x), max(len(references), int(round(len(x) * fraction))))
    grid = energy_grid(config)
    for output_index in range(len(x) - count, len(x)):
        base, labels = references[int(rng.integers(len(references)))]
        shift = rng.uniform(-0.018, 0.018)
        shifted = np.interp(grid + shift, grid, base, left=base[0], right=base[-1])
        scale = 10.0 ** rng.uniform(-0.30, 0.30)
        tilt = 1.0 + rng.uniform(-0.08, 0.08) * (grid - grid.mean()) / np.ptp(grid)
        expected = np.maximum(shifted * scale * tilt, 0.0)
        x[output_index] = rng.poisson(expected).astype(np.float32)
        y[output_index] = labels
    order = rng.permutation(len(x))
    return x[order], y[order]


def _save_dataset(path: str, config: SpectrumConfig, samples: int, seed: int,
                  example_csv: str = "", backgrounds=(), labeled_references=(),
                  real_fraction: float = 0.2):
    templates = _background_templates(backgrounds, config)
    energy, x, y = generate_dataset(config, samples, seed, templates)
    references = _parse_labeled_references(labeled_references, config)
    x, y = _add_real_augmentations(x, y, references, real_fraction, seed, config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    training_metadata = {
        "background_reference_count": len(backgrounds),
        "labeled_reference_count": len(labeled_references),
        "real_augmentation_fraction": real_fraction if labeled_references else 0.0,
    }
    np.savez_compressed(path, energy=energy, counts=x, labels=y,
                        elements=np.asarray(config.elements), config=json.dumps(config.to_dict()),
                        training_metadata=json.dumps(training_metadata))
    print(f"Dataset created: {path} ({samples} spectra, {len(config.elements)} elements)")
    if example_csv:
        Path(example_csv).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(example_csv, np.column_stack((energy, x[0])), delimiter=",",
                   header="Energy,Counts", comments="", fmt="%.6g")
        present = [e for e, flag in zip(config.elements, y[0]) if flag]
        print(f"Example CSV: {example_csv} (ground truth: {', '.join(present)})")


def _load_dataset(path: str):
    data = np.load(path, allow_pickle=False)
    return data, SpectrumConfig.from_dict(json.loads(str(data["config"])))


def _train(dataset: str, output: str, seed: int):
    data, config = _load_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        data["counts"], data["labels"], test_size=0.2, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed
    )
    model_config = config.to_dict()
    if "training_metadata" in data.files:
        model_config["training_metadata"] = json.loads(str(data["training_metadata"]))
    model = fit_model(x_train, y_train, x_val, y_val, data["elements"].tolist(), model_config)
    report = metrics(y_test, model.predict_proba(x_test), model.thresholds)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    Path(output + ".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Model saved: {output}")


def _simulate_with_config(config: SpectrumConfig, elements_text: str, output: str, plot: str, seed: int):
    requested = [item.strip() for item in elements_text.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(config.elements))
    if unknown:
        raise ValueError(f"Unsupported elements: {', '.join(unknown)}")
    if not requested:
        raise ValueError("Specify at least one element")
    selected = np.asarray([config.elements.index(element) for element in requested])
    energy, counts, _ = simulate_spectrum(config, np.random.default_rng(seed), selected)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.column_stack((energy, counts)), delimiter=",",
               header="Energy,Counts", comments="", fmt="%.6g")
    truth_path = output + ".truth.json"
    Path(truth_path).write_text(json.dumps({"elements": requested}, indent=2) + "\n")
    if plot:
        plot_spectrum(energy, counts, plot, requested,
                      title=f"Synthetic spectrum ({config.anode_material} anode) — ground truth: {', '.join(requested)}")
    print(f"Spectrum: {output}")
    print(f"Ground truth: {truth_path} ({', '.join(requested)})")
    if plot:
        print(f"Plot: {plot}")


def _predict(model_path: str, spectrum_path: str, show_all: bool, plot: str):
    model = joblib.load(model_path)
    config = SpectrumConfig.from_dict(model.config)
    energy, counts = read_spectrum(spectrum_path, config)
    probabilities = model.predict_proba(counts)[0]
    rows = []
    for element, probability, threshold in zip(model.elements, probabilities, model.thresholds):
        present = bool(probability >= threshold)
        if present or show_all:
            rows.append({"element": element, "probability": round(float(probability), 4),
                         "threshold": round(float(threshold), 4), "present": present})
    result = {"spectrum": spectrum_path, "predictions": sorted(rows, key=lambda r: -r["probability"]),
              "peak_scores": peak_scores(energy, counts, config)}
    if plot:
        detected = [row["element"] for row in rows if row["present"]]
        probability_map = {element: float(probability)
                           for element, probability in zip(model.elements, probabilities)}
        plot_spectrum(energy, counts, plot, detected, "XRF spectrum — predicted elements", probability_map)
        result["plot"] = plot
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="FluoX-Mo: multi-label XRF identification")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate a synthetic dataset")
    generate.add_argument("--output", default="data/synthetic.npz")
    generate.add_argument("--samples", type=int, default=5000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--anode", choices=("Cu", "Mo", "Ag"), default="Mo")
    generate.add_argument("--example-csv", default="", help="Also export the first spectrum as CSV")
    generate.add_argument(
        "--background", action="append", default=[], metavar="SPECTRUM",
        help="Use a real spectrum to derive the background (repeatable)",
    )
    generate.add_argument(
        "--labeled-reference", action="append", default=[], metavar="PATH=ELEMENTS",
        help="Add augmented real spectra with known labels (repeatable)",
    )
    generate.add_argument(
        "--real-fraction", type=float, default=0.2,
        help="Fraction replaced by labeled-real augmentations (default: 0.2)",
    )
    train = sub.add_parser("train", help="Train and evaluate the model")
    train.add_argument("--dataset", default="data/synthetic.npz")
    train.add_argument("--output", default="models/fluox.joblib")
    train.add_argument("--seed", type=int, default=42)
    simulate = sub.add_parser("simulate", help="Create a spectrum with known composition")
    simulate.add_argument("--elements", required=True, help="Example: Fe,Ni,Cr")
    simulate.add_argument("--output", default="data/test_spectrum.csv")
    simulate.add_argument("--plot", default="outputs/test_spectrum.png")
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--anode", choices=("Cu", "Mo", "Ag"), default="Mo")
    predict = sub.add_parser("predict", help="Analyze an Energy, Counts file")
    predict.add_argument("spectrum")
    predict.add_argument("--model", default="models/fluox.joblib")
    predict.add_argument("--all", action="store_true", help="Also show elements classified as absent")
    predict.add_argument("--plot", default="", help="Save an annotated PNG plot")
    args = parser.parse_args()
    if args.command == "generate":
        _save_dataset(args.output, SpectrumConfig(anode_material=args.anode), args.samples,
                      args.seed, args.example_csv, args.background,
                      args.labeled_reference, args.real_fraction)
    elif args.command == "train":
        _train(args.dataset, args.output, args.seed)
    elif args.command == "simulate":
        config = SpectrumConfig(anode_material=args.anode)
        _simulate_with_config(config, args.elements, args.output, args.plot, args.seed)
    else:
        _predict(args.model, args.spectrum, args.all, args.plot)


if __name__ == "__main__":
    main()
