"""Generated-sample helpers: pitch estimate, repitch, trim (no audio device, no network)."""
import unittest

import numpy as np

from outer_vision.pitch import estimate_f0, repitch, trim

SR = 24000


def note(f, secs=1.0, harmonics=(1.0, 0.8, 0.5, 0.3), attack_silence=0.2, sr=SR):
    t = np.arange(int(secs * sr)) / sr
    w = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harmonics)) * np.exp(-t / 0.6)
    w = np.concatenate([np.zeros(int(attack_silence * sr)), w])
    return (w / np.abs(w).max() * 20000).astype(np.int16)


class TestPitch(unittest.TestCase):
    def test_f0_of_harmonic_notes(self):
        for f in (130.8, 261.63, 440.0, 523.25):
            self.assertAlmostEqual(estimate_f0(note(f), SR), f, delta=f * 0.01)

    def test_strong_second_harmonic_is_not_an_octave_error(self):
        self.assertAlmostEqual(estimate_f0(note(261.63, harmonics=(0.4, 1.0, 0.3)), SR), 261.63, delta=3)

    def test_noise_is_unpitched(self):
        x = (np.random.default_rng(0).normal(size=SR) * 5000).astype(np.int16)
        self.assertIsNone(estimate_f0(x, SR))

    def test_repitch_moves_f0(self):
        x = note(261.63).astype(np.float32)
        up = repitch(x, 440.0 / 261.63)
        self.assertAlmostEqual(estimate_f0((up * 1).astype(np.int16), SR), 440.0, delta=5)

    def test_trim_drops_leading_silence(self):
        x = trim(note(261.63, attack_silence=0.5), SR)
        self.assertLess(np.argmax(np.abs(x) > 1000) / SR, 0.01)
        self.assertLessEqual(np.abs(x).max(), 32767 * 0.9 + 1)


if __name__ == "__main__":
    unittest.main()
