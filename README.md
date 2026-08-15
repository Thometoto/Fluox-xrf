# FluoX-Mo — Automated Element Identification from XRF Spectra

V1 prototype for a single-point acquisition:

```text
Energy (keV), Counts → preprocessing → multi-label model → probability per element
```

The project works without experimental data by using a configurable simulator. Results
obtained from synthetic data validate the software pipeline; they are **not an experimental
validation of the FluoX instrument**.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Complete demonstration

```bash
fluox-mo generate --samples 5000 --output data/synthetic.npz
fluox-mo train --dataset data/synthetic.npz --output models/fluox.joblib
fluox-mo predict path/to/spectrum.csv --model models/fluox.joblib --all
```

## Visual test with a known composition

```bash
fluox-mo simulate --elements Fe,Ni,Cr \
  --output data/test_fe_ni_cr.csv \
  --plot outputs/test_fe_ni_cr.png

fluox-mo predict data/test_fe_ni_cr.csv \
  --model models/fluox.joblib \
  --plot outputs/prediction_fe_ni_cr.png
```

The first plot marks the ground-truth lines. The second marks only the predicted elements
and displays their probabilities. The `*.truth.json` file enables automatic comparison.

## Input format

A real input file must contain two numeric columns, with or without a header:

```csv
Energy,Counts
2.000,12
2.020,15
```

It must cover the full model energy range, which is 2–26 keV by default.

## V1 elements

Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, As, Se, Rb, Sr, Y, Zr, Pb, Mo, and Ag,
except for the material used as the tube anode.

The selected tube-anode element is intentionally excluded. Its sample emission lines overlap
with lines scattered from the source. Identifying it requires a dedicated instrument protocol
or an additional reference measurement; reporting a simple probability would be misleading.

Provisional instrument parameters are centralized in `src/fluox/config.py`. The simulator
models the main emission lines, energy-dependent detector resolution, continuous background,
Poisson counting statistics, small calibration errors, and scattering from the Mo anode.

## Scientific limitations

- Relative intensities and matrix effects are deliberately simplified.
- Mo in a sample is difficult to distinguish from scattered Mo anode lines.
- Thresholds learned from synthetic data are not instrument detection limits.
- A model intended for real measurements must be adapted and validated with standards
  measured on FluoX.

The `*.metrics.json` file generated during training reports micro/macro F1, precision,
recall, and exact match on a held-out synthetic test set.

## Command-line use (without the web interface)

Select the tube anode used during acquisition. The script automatically loads the matching
model and writes an annotated plot and a JSON report to `outputs/`.

```bash
python scripts/analyze_spectrum.py spectrum.csv --anode Mo
```

Complete examples using the three bundled files:

```bash
python scripts/analyze_spectrum.py data/example_cu.csv --anode Cu
python scripts/analyze_spectrum.py data/example_mo.csv --anode Mo
python scripts/analyze_spectrum.py data/example_ag.csv --anode Ag
```

Display every model output, including absent elements:

```bash
python scripts/analyze_spectrum.py data/example_mo.csv --anode Mo --show-all
```

View the built-in help at any time:

```bash
python scripts/analyze_spectrum.py --help
```

## Local web interface (recommended)

```bash
python webapp.py
```

Then open `http://127.0.0.1:5000`. Files are analyzed locally and are not uploaded to the
Internet.

On macOS, you can also double-click `Launch-FluoX.command`. The browser opens automatically.
The interface previews the spectrum before analysis and displays a progress bar until the
results appear.

### Tube-anode selection

The web interface supports Cu, Mo, and Ag tube anodes. Each option loads a separately trained
model containing the appropriate scattered source lines:

- Cu Kα/Kβ: approximately 8.048/8.905 keV;
- Mo Kα/Kβ: approximately 17.479/19.608 keV;
- Ag Kα/Kβ: approximately 22.163/24.942 keV.

Select the anode that was actually used for the acquisition. The corresponding anode element
is excluded from the output because it cannot be distinguished reliably from source scattering.
