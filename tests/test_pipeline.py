import numpy as np

from fluox.baseline import multi_line_evidence, peak_scores
from fluox.config import SpectrumConfig
from fluox.interpretation import interpret_predictions
from fluox.preprocessing import energy_reliability, estimate_background, extract_features, transform_counts
from fluox.simulator import generate_dataset, simulate_spectrum


def test_dataset_is_reproducible_and_multilabel():
    config = SpectrumConfig(channels=256)
    assert config.energy_min_kev == 1.0
    assert config.energy_max_kev == 20.7
    assert SpectrumConfig().channels == 4096
    assert {"Si", "P", "S", "Cl", "Ar", "K", "Ca", "Br", "Ba"}.issubset(config.elements)
    e1, x1, y1 = generate_dataset(config, 12, seed=7)
    e2, x2, y2 = generate_dataset(config, 12, seed=7)
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert x1.shape == (12, 256)
    assert np.all(y1.sum(axis=1) >= 1)
    assert np.all(y1.sum(axis=1) <= 16)


def test_preprocessing_is_finite_and_normalized():
    transformed = transform_counts(np.array([[0, 1, 10], [10, 0, 3]], dtype=float))
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=1e-6)


def test_full_spectrum_features_are_finite_and_keep_local_information():
    config = SpectrumConfig(channels=256)
    counts = np.ones((2, config.channels), dtype=float)
    counts[:, 80:84] = 100.0
    background = estimate_background(counts, config)
    features = extract_features(counts, config)
    reliability = energy_reliability(config)
    assert background.shape == counts.shape
    assert features.shape[0] == 2
    assert features.shape[1] > config.channels // 2
    assert np.isfinite(features).all()
    assert reliability.shape == (config.channels,)
    assert reliability.min() < reliability.max()


def test_peak_baseline_finds_strong_iron():
    config = SpectrumConfig(total_counts_min=1_000_000, total_counts_max=1_000_001)
    iron_index = np.array([config.elements.index("Fe")])
    energy, counts, labels = simulate_spectrum(config, np.random.default_rng(2), iron_index)
    scores = peak_scores(energy, counts, config)
    assert labels[config.elements.index("Fe")] == 1
    assert scores["Fe"] > 3.0


def test_multi_line_evidence_is_finite():
    config = SpectrumConfig(channels=256, total_counts_min=1_000_000,
                            total_counts_max=1_000_001)
    iron_index = np.array([config.elements.index("Fe")])
    _, counts, _ = simulate_spectrum(config, np.random.default_rng(3), iron_index)
    evidence = multi_line_evidence(counts, config)
    assert set(evidence) == set(config.elements)
    assert evidence["Fe"]["supported_lines"] >= 1
    assert np.isfinite([row["max_snr"] for row in evidence.values()]).all()


def test_interpretation_marks_argon_as_atmospheric():
    config = SpectrumConfig(channels=256)

    class DummyModel:
        elements = config.elements
        thresholds = np.full(len(config.elements), 0.5)

        def predict_proba(self, counts):
            scores = np.zeros((1, len(self.elements)))
            scores[0, self.elements.index("Ar")] = 0.9
            return scores

    rows = interpret_predictions(DummyModel(), config, np.ones(config.channels))
    argon = next(row for row in rows if row["element"] == "Ar")
    assert argon["active"]
    assert argon["status"] == "atmospheric"
    assert argon["primary_line_type"] == "Kα"


def test_lead_uses_its_strongest_l_line_for_display():
    config = SpectrumConfig(channels=256)

    class DummyModel:
        elements = config.elements
        thresholds = np.full(len(config.elements), 0.5)

        def predict_proba(self, counts):
            scores = np.zeros((1, len(self.elements)))
            scores[0, self.elements.index("Pb")] = 0.9
            return scores

    rows = interpret_predictions(DummyModel(), config, np.ones(config.channels))
    lead = next(row for row in rows if row["element"] == "Pb")
    assert lead["primary_line_type"] == "Lβ"
    assert lead["primary_energy"] == 12.614
