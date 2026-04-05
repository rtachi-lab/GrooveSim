from __future__ import annotations

import unittest

import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from groovesim.features import (
    compute_idyom_like_surprisal,
    compute_syncopation_index,
    estimate_meter_from_grid,
)
from groovesim.pipeline import analyze_audio_bytes, analyze_audio_file, analyze_midi_bytes, analyze_midi_file, analyze_onset_file
from groovesim.webapp import app


class PipelineSmokeTests(unittest.TestCase):
    def test_analyze_onsets_smoke(self) -> None:
        result = analyze_onset_file("examples/simple_onsets.json")
        self.assertEqual(result["source"], "onsets")
        self.assertGreaterEqual(result["groove_score"], 0.0)
        self.assertLessEqual(result["groove_score"], 1.0)
        self.assertGreater(result["tempo_bpm"], 0.0)

    def test_analyze_audio_smoke(self) -> None:
        result = analyze_audio_file("examples/pulse_train.wav")
        self.assertEqual(result["source"], "audio")
        self.assertGreaterEqual(result["groove_score"], 0.0)
        self.assertLessEqual(result["groove_score"], 1.0)
        self.assertGreater(result["tempo_bpm"], 0.0)

    def test_analyze_midi_smoke(self) -> None:
        result = analyze_midi_file("examples/simple_pattern.mid")
        self.assertEqual(result["source"], "midi")
        self.assertGreaterEqual(result["groove_score"], 0.0)
        self.assertLessEqual(result["groove_score"], 1.0)
        self.assertGreater(result["tempo_bpm"], 0.0)

    def test_analyze_audio_bytes_smoke(self) -> None:
        result = analyze_audio_bytes(Path("examples/pulse_train.wav").read_bytes())
        self.assertEqual(result["source"], "audio")
        self.assertGreater(result["tempo_bpm"], 0.0)

    def test_analyze_audio_bytes_with_tempo_prior(self) -> None:
        result = analyze_audio_bytes(Path("examples/pulse_train.wav").read_bytes(), tempo_prior=115.0)
        self.assertEqual(result["source"], "audio")
        self.assertGreaterEqual(result["tempo_bpm"], 110.0)
        self.assertLessEqual(result["tempo_bpm"], 120.0)

    def test_analyze_midi_bytes_smoke(self) -> None:
        result = analyze_midi_bytes(Path("examples/simple_pattern.mid").read_bytes())
        self.assertEqual(result["source"], "midi")
        self.assertGreater(result["tempo_bpm"], 0.0)

    def test_meter_aware_syncopation_prefers_offbeat_pattern(self) -> None:
        onbeat = np.array([1, 0, 0, 0] * 4, dtype=int)
        offbeat = np.array([1, 0, 0, 1, 0, 0, 0, 0] * 2, dtype=int)
        meter_on = estimate_meter_from_grid(onbeat, candidate_steps_per_beat=(4,), candidate_beats_per_bar=(4,))
        meter_off = estimate_meter_from_grid(offbeat, candidate_steps_per_beat=(4,), candidate_beats_per_bar=(4,))
        self.assertLess(compute_syncopation_index(meter_on), compute_syncopation_index(meter_off))

    def test_idyom_like_surprisal_penalizes_irregular_pattern(self) -> None:
        regular = np.array([1, 0, 0, 0] * 8, dtype=int)
        irregular = np.array([1, 0, 1, 0, 0, 1, 0, 0] * 4, dtype=int)
        reg_meter = estimate_meter_from_grid(regular, candidate_steps_per_beat=(4,), candidate_beats_per_bar=(4,))
        irr_meter = estimate_meter_from_grid(irregular, candidate_steps_per_beat=(4,), candidate_beats_per_bar=(4,))
        reg = compute_idyom_like_surprisal(regular, meter=reg_meter)
        irr = compute_idyom_like_surprisal(irregular, meter=irr_meter)
        self.assertLess(reg["mean_surprisal"], irr["mean_surprisal"])

    def test_web_api_accepts_tempo_hint_form_field(self) -> None:
        client = TestClient(app)
        with Path("examples/pulse_train.wav").open("rb") as handle:
            response = client.post(
                "/api/analyze",
                files={"file": ("pulse_train.wav", handle, "audio/wav")},
                data={"tempo_hint": "115"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tempo_hint"], 115.0)
        self.assertGreaterEqual(payload["tempo_bpm"], 110.0)
        self.assertLessEqual(payload["tempo_bpm"], 120.0)


if __name__ == "__main__":
    unittest.main()
