"""Fast regression tests for the Stage-4 physics and statistics.

These tests use the Python standard-library test runner, read no filterbank,
and require no GPU:

    python -m unittest -v test_stage4_core.py
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_preprocessing import (  # noqa: E402
    CandidateParams,
    dechirp,
    frequency_integrate,
    make_candidate_view,
    track_channels,
)
from channelized_injection import (  # noqa: E402
    broadened_template,
    channel_integrated_weights,
    inject_broadened_signal,
    safe_center_channel_interval,
)

DF_HZ = 2.980232239
DT_S = 0.67108864


def synthetic_track(shape, params, amplitude=10.0):
    out = np.zeros(shape, dtype=np.float64)
    for row, centre in enumerate(track_channels(params, shape[0])):
        lo = int(np.floor(centre))
        frac = centre - lo
        if 0 <= lo < shape[1]:
            out[row, lo] += amplitude * (1.0 - frac)
        if 0 <= lo + 1 < shape[1]:
            out[row, lo + 1] += amplitude * frac
    return out


class Stage4CoreTests(unittest.TestCase):
    def test_pair_factory_all_cases_on_signed_mock_filterbank(self):
        rng = np.random.default_rng(7)
        background_a = rng.normal(size=(16, 16384)).astype(np.float32)
        background_b = rng.normal(size=(16, 16384)).astype(np.float32)
        fake_train = types.ModuleType("train")
        fake_train.CONFIGS = {"high_freq": {"frame_shape": (16, 1024)}}

        def load_background(path):
            data = background_a if path == "a" else background_b
            return data, {
                "foff": -2.980232239e-6,
                "tsamp": DT_S,
            }

        fake_train.load_background = load_background
        previous_train = sys.modules.get("train")
        previous_data = sys.modules.pop("stage4_data", None)
        sys.modules["train"] = fake_train
        try:
            from stage4_data import CandidatePairFactory

            for split in ("train", "val", "test"):
                factory = CandidatePairFactory(
                    "high_freq",
                    "a",
                    "b",
                    station_b_is_proxy=True,
                    split=split,
                )
                for index, case in enumerate(
                    ("match", "onesided", "independent", "noise")
                ):
                    item = factory.sample(
                        100 + index,
                        fixed_width_hz=100.0,
                        fixed_shape="lorentzian",
                        fixed_case=case,
                        return_raw=True,
                    )
                    self.assertEqual(item["view_a"].shape, (16, 1024))
                    self.assertTrue(np.all(np.isfinite(item["view_a"])))
                    self.assertEqual(item["label"], int(case == "match"))
                    self.assertTrue(item["record"].station_b_proxy)
        finally:
            sys.modules.pop("stage4_data", None)
            if previous_data is not None:
                sys.modules["stage4_data"] = previous_data
            if previous_train is None:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = previous_train

    def test_pair_factory_contains_tracks_at_reported_lofts_cadence(self):
        """Regression for tsamp=18.119 s station-B truncation in full training."""

        rng = np.random.default_rng(71)
        background_a = rng.normal(size=(16, 16384)).astype(np.float32)
        background_b = rng.normal(size=(16, 16384)).astype(np.float32)
        fake_train = types.ModuleType("train")
        fake_train.CONFIGS = {"high_freq": {"frame_shape": (16, 1024)}}

        def load_background(path):
            data = background_a if path == "a" else background_b
            return data, {
                # Exact values reported for LOFTS0192.rawspec.0000.fil.
                "foff": 2.9729986653744596e-6,
                "tsamp": 18.11939328,
            }

        fake_train.load_background = load_background
        previous_train = sys.modules.get("train")
        previous_data = sys.modules.pop("stage4_data", None)
        sys.modules["train"] = fake_train
        try:
            from stage4_data import CandidatePairFactory

            factory = CandidatePairFactory(
                "high_freq",
                "a",
                "b",
                station_b_is_proxy=True,
                split="train",
            )
            # Seed 8 failed before this fix (station-B captured fraction 0.8314).
            # A run of nearby seeds guards against reintroducing stochastic edge
            # clipping for the widest, longest-wing profile.
            for seed in range(8, 20):
                item = factory.sample(
                    seed,
                    fixed_width_hz=100.0,
                    fixed_shape="lorentzian",
                    fixed_case="independent",
                )
                record = item["record"]
                self.assertEqual(record.true_reference_row, 7.5)
                self.assertGreaterEqual(record.in_tile_energy_fraction_min_a, 0.95)
                self.assertGreaterEqual(record.in_tile_energy_fraction_min_b, 0.95)
                self.assertTrue(np.all(np.isfinite(item["view_a"])))
                self.assertTrue(np.all(np.isfinite(item["view_b"])))
        finally:
            sys.modules.pop("stage4_data", None)
            if previous_data is not None:
                sys.modules["stage4_data"] = previous_data
            if previous_train is None:
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = previous_train

    def test_safe_center_interval_matches_injector_mass_definition(self):
        for profile in ("lorentzian", "gaussian", "box"):
            with self.subTest(profile=profile):
                lo, hi = safe_center_channel_interval(
                    1024,
                    width_hz=100.0,
                    df_hz=2.9729986653744596,
                    shape=profile,
                )
                for center in (lo, (lo + hi) / 2.0, hi):
                    params = CandidateParams(
                        center,
                        0.0,
                        100.0,
                        2.9729986653744596,
                        18.11939328,
                        reference_row=7.5,
                    )
                    _, captured = broadened_template(
                        (16, 1024), params, profile_shape=profile
                    )
                    self.assertGreaterEqual(captured, 0.95)

    def test_dechirp_obeys_signed_frequency_axis(self):
        for signed_step in (DF_HZ, -DF_HZ):
            with self.subTest(signed_step=signed_step):
                params = CandidateParams(100.0, 3.2, 10.0, signed_step, DT_S)
                signal = synthetic_track((32, 256), params)
                corrected = dechirp(
                    signal,
                    params.drift_hz_per_s,
                    params.channel_step_hz,
                    params.dt_s,
                )
                wrong = dechirp(
                    signal,
                    params.drift_hz_per_s,
                    -params.channel_step_hz,
                    params.dt_s,
                )
                self.assertLess(
                    float(np.std(np.argmax(corrected, axis=1))),
                    0.75,
                )
                self.assertGreater(
                    float(np.std(np.argmax(wrong, axis=1))),
                    2.0,
                )

    def test_dechirp_zero_padding_does_not_replicate_edge_rfi(self):
        data = np.zeros((8, 32), dtype=np.float64)
        data[:, -1] = 7.0
        shifted, validity = dechirp(
            data,
            drift_hz_per_s=-3.0,
            channel_step_hz=1.0,
            dt_s=1.0,
            return_validity=True,
        )
        self.assertTrue(np.all(shifted[validity < 0.5] == 0.0))
        self.assertLessEqual(
            int(np.count_nonzero(shifted)),
            int(np.count_nonzero(data)),
        )
        self.assertLessEqual(float(np.max(shifted)), 7.0)

    def test_frequency_filter_preserves_white_noise_scale(self):
        rng = np.random.default_rng(123)
        noise = rng.normal(size=(128, 2048))
        for kind in ("boxcar", "lorentzian", "gaussian"):
            with self.subTest(kind=kind):
                filtered = frequency_integrate(
                    noise,
                    width_hz=30.0,
                    df_hz=DF_HZ,
                    kind=kind,
                )
                sigma = float(np.std(filtered[:, 200:-200]))
                self.assertGreater(sigma, 0.90)
                self.assertLess(sigma, 1.10)

    def test_channel_integrated_profiles_conserve_energy(self):
        for shape in ("lorentzian", "box", "gaussian"):
            with self.subTest(shape=shape):
                weights, captured = channel_integrated_weights(
                    1024,
                    center_channel=511.5,
                    width_hz=100.0,
                    df_hz=DF_HZ,
                    shape=shape,
                )
                self.assertTrue(np.isclose(weights.sum(), 1.0, atol=1e-12))
                self.assertGreater(captured, 0.95)
                self.assertTrue(np.all(weights >= 0.0))

    def test_broadened_template_rejects_edge_truncation(self):
        params = CandidateParams(1.0, 0.0, 100.0, -DF_HZ, DT_S)
        with self.assertRaisesRegex(ValueError, "truncated"):
            broadened_template((16, 1024), params)

    def test_detected_and_power_snr_are_distinct(self):
        rng = np.random.default_rng(8128)
        background = rng.normal(size=(16, 1024))
        p10 = CandidateParams(511.5, -0.5, 10.0, -DF_HZ, DT_S)
        p100 = CandidateParams(511.5, -0.5, 100.0, -DF_HZ, DT_S)
        _, _, d10 = inject_broadened_signal(
            background,
            p10,
            target_snr=12.0,
            snr_mode="detected",
            rng=np.random.default_rng(2),
        )
        _, _, d100 = inject_broadened_signal(
            background,
            p100,
            target_snr=12.0,
            snr_mode="detected",
            rng=np.random.default_rng(2),
        )
        _, _, q10 = inject_broadened_signal(
            background,
            p10,
            target_snr=12.0,
            snr_mode="power",
            rng=np.random.default_rng(2),
        )
        _, _, q100 = inject_broadened_signal(
            background,
            p100,
            target_snr=12.0,
            snr_mode="power",
            rng=np.random.default_rng(2),
        )
        self.assertAlmostEqual(d10.expected_broadened_snr, 12.0, places=10)
        self.assertAlmostEqual(d100.expected_broadened_snr, 12.0, places=10)
        self.assertGreater(d100.amplitude, d10.amplitude)
        self.assertLess(
            q100.expected_broadened_snr,
            q10.expected_broadened_snr,
        )

    def test_candidate_view_is_finite_and_recentred(self):
        rng = np.random.default_rng(991)
        params = CandidateParams(350.25, 1.2, 30.0, -DF_HZ, DT_S)
        template, _ = broadened_template((16, 1024), params, profile_shape="lorentzian")
        signal = 40.0 * template
        raw = rng.normal(size=signal.shape) + signal
        view = make_candidate_view(raw, params, integration="lorentzian")
        self.assertEqual(view.shape, raw.shape)
        self.assertTrue(np.all(np.isfinite(view)))
        centre = view.shape[1] // 2
        peak = int(np.argmax(np.mean(view, axis=0)))
        self.assertLessEqual(abs(peak - centre), 1)

    def test_paired_bootstrap_identical_methods_have_zero_delta(self):
        try:
            from evaluate_stage4 import (
                METHODS,
                paired_stratified_bootstrap,
            )
        except ImportError as exc:
            self.skipTest("optional evaluation dependencies unavailable: %s" % exc)
        labels = np.asarray([0] * 40 + [1] * 40)
        base = np.linspace(-1.0, 1.0, labels.size)
        scores = {method: base.copy() for method in METHODS}
        strata = np.asarray(["negative"] * 40 + ["positive"] * 40)
        result = paired_stratified_bootstrap(
            labels,
            scores,
            strata,
            n_boot=200,
            ci_level=0.95,
            seed=44,
        )
        for delta in result["deltas"].values():
            self.assertAlmostEqual(delta["delta_auc"], 0.0)
            self.assertAlmostEqual(delta["ci_lo"], 0.0)
            self.assertAlmostEqual(delta["ci_hi"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
