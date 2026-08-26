# FluoX — Automated Element Identification from XRF Spectra

V1 prototype for a single-point acquisition:

```text
Energy (keV), Counts → preprocessing → multi-label model → score per element
```

The project uses a configurable simulator. The current Mo/air model can derive realistic
background shapes from laboratory reference spectra, then inject varied synthetic elemental
signatures. The current model has also been adapted with five labeled Mo/air laboratory
spectra. This improves realism but is not, by itself, independent experimental validation.

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
and displays their model scores. The `*.truth.json` file enables automatic comparison.

## Input format

A real input file must contain two numeric columns, with or without a header:

```csv
Energy,Counts
1.000,12
1.020,15
```

It must cover the model energy range, currently 1–20.7 keV. The complete uploaded
acquisition is displayed. Channels above 16 keV remain available to the model, but a
physics-informed reliability prior reduces the influence of the Mo Compton and elastic regions.

## V1 elements

Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga,
As, Se, Br, Rb, Sr, Y, Zr, Nb, Pb, Ba, and Mo,
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

The displayed score is the sigmoid output of an independent one-vs-rest logistic classifier.
It is not an experimentally calibrated probability. For each element, the decision threshold
is selected from 0.10 to 0.90 to maximize F1 on validation data. It is therefore a model
decision rule, not a universal confidence level or an instrumental detection limit.

The classifier uses the complete 4,096-channel acquisition through a physics-informed feature
pipeline: logarithmic counts, an estimated smooth background, the background-corrected
residual, local evidence around each expected emission line, and a soft reliability weighting
around the Mo Compton/elastic regions. No energy channel is simply discarded.

The background is estimated with a nonlinear SNIP-style procedure, so the analysis does not
assume a straight baseline. Expected K or L lines are then inspected separately to provide
an interpretable local peak-evidence indicator. This evidence qualifies the classifier result;
it does not create a detection by itself because line windows overlap in crowded spectra.

Results are reported as **detected**, **possible trace**, **ambiguous**, **uncertain**, or
**air signal**. Argon is identified as an atmospheric contribution. Arsenic is flagged as
ambiguous when Pb or Br is also active, and Ba is flagged when Ti can explain the same region.
Lead is modeled using its L lines.

### Experimental cross-validation

Four closely related clay spectra were evaluated with leave-one-spectrum-out validation: each
file was excluded from adaptation in turn and predicted by a newly trained model. Across the
four held-out files, 69 of 70 expected element labels were recovered (98.6% recall) with no
additional confirmed element (100% precision). Sulfur in one held-out clay was the only miss.
These results are encouraging but remain preliminary because the validation set is small and
the four matrices are very similar.

The `*.metrics.json` file generated during training reports micro/macro F1, precision,
recall, and exact match on a held-out synthetic test set.

## Command-line use (without the web interface)

Select the tube anode used during acquisition. The script automatically loads the matching
model and writes an annotated plot and a JSON report to `outputs/`.

```bash
python scripts/analyze_spectrum.py spectrum.csv --anode Mo
```

Complete example using the currently supported Mo/air profile:

```bash
python scripts/analyze_spectrum.py data/example_mo.csv --anode Mo
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

The current application profile supports a **Mo tube anode with measurements in air**.
Cu/Ag anodes and vacuum measurements remain visible but disabled until experimental
reference data are available. Argon is reported as an atmospheric signal. A detected Y or
Zr signal is marked uncertain because its main usable line lies in the rising Mo-scattering
region near the upper analysis boundary.

The model uses a fixed 4,096-channel grid between 1 and 20.7 keV. The graph displays every
original measured point, including points outside the internal interpolation grid.

Reference-derived synthetic training data can be generated locally without committing the
private laboratory files:

```bash
fluox-mo generate --anode Mo --samples 5000 \
  --background path/to/reference_1.csv \
  --background path/to/reference_2.csv \
  --labeled-reference 'path/to/reference_1.csv=Si,P,Ar,K,Ca,Fe' \
  --real-fraction 0.2 \
  --output data/synthetic_mo_air.npz
```

Labeled-reference augmentation is an adaptation step, not an independent validation. Source
spectra used this way must not also be presented as unseen test data; final performance must be
measured on separate experimental samples.

### Tube-anode selection

The web interface supports Cu, Mo, and Ag tube anodes. Each option loads a separately trained
model containing the appropriate scattered source lines:

- Cu Kα/Kβ: approximately 8.048/8.905 keV;
- Mo Kα/Kβ: approximately 17.479/19.608 keV, explicitly modeled as source artefacts;
- Ag Kα/Kβ: approximately 22.163/24.942 keV, outside the current model grid.

Select the anode that was actually used for the acquisition. The corresponding anode element
is excluded from the output because it cannot be distinguished reliably from source scattering.
