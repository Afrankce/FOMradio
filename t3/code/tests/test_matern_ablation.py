import copy
import unittest

import numpy as np
from scipy.special import k1
import torch

from evaluation.matern_ablation_metrics import haar_step, scene_diagnostics, validation_gate
from model.matern_multiscale_uno import MaternBankSampler, MaternMultiscaleUNO2d, embedding_spectra


class MaternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_psd_and_covariance_reconstruction(self):
        for level in range(5):
            size = 256 // 2 ** level
            extent, amplitudes, minimum = embedding_spectra(size, level)
            self.assertGreaterEqual(minimum, -1e-10)
            for schedule in range(2):
                for j, length in enumerate((2., 4., 8.)):
                    length *= 2 ** -level if schedule == 0 else 1
                    covariance = np.fft.irfft2(amplitudes[schedule, j].astype(float) ** 2, s=(extent, extent))
                    self.assertAlmostEqual(covariance[0, 0], 1., places=5)
                    self.assertAlmostEqual(covariance[0, 1], k1(1 / length) / length, places=5)

    def test_seed_frozen_unit_variance_and_spatial_correlation(self):
        metadata = [dict(scene_id=f"u{i}", steering_deg=0) for i in range(200)]
        state = torch.get_rng_state().clone()
        sampler = MaternBankSampler("M2", resolution=32)
        a = sampler.sample(metadata, device="cpu", namespace="unit")
        b = sampler.sample(metadata, device="cpu", namespace="unit")
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertLess(abs(float(a[0].mean())), .1)
        self.assertLess(abs(float(a[0].square().mean()) - 1), .15)
        observed = (a[0][:, 0, :, :-1] * a[0][:, 0, :, 1:]).mean().item()
        self.assertAlmostEqual(observed, k1(.5) * .5, delta=.12)
        white = MaternBankSampler("B2", 32).sample(metadata, device="cpu", namespace="unit")
        self.assertLess(abs(float((white[0][..., :-1] * white[0][..., 1:]).mean())), .03)

    def test_initialization_feedback_gradients_and_missing_bank(self):
        torch.manual_seed(42)
        fixed = MaternMultiscaleUNO2d("M2", operator_width=4, operator_modes=(4, 4, 4, 2, 1))
        torch.manual_seed(42)
        feedback = MaternMultiscaleUNO2d("M3", operator_width=4, operator_modes=(4, 4, 4, 2, 1))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(fixed.state_dict().values(), feedback.state_dict().values())))
        bank = MaternBankSampler("M2", 64).sample([dict(scene_id="u1", steering_deg=0)], device="cpu", namespace="test")
        x, c = torch.randn(1, 1, 64, 64), torch.randn(1, 3, 64, 64)
        fixed.eval()
        feedback.eval()
        with self.assertRaises(RuntimeError):
            fixed(image=c, x=x, step=.2)
        with fixed.use_innovations(bank):
            expected = fixed(image=c, x=x, step=.2)
            self.assertTrue(torch.equal(expected, fixed(image=c, x=x, step=.2)))
        with feedback.use_innovations(bank):
            prediction = feedback(image=c, x=x, step=.2)
            self.assertTrue(torch.equal(expected, prediction))
            prediction.square().mean().backward()
        for injection in feedback.injections:
            self.assertTrue(torch.isfinite(injection.head[-1].weight.grad).all())
            self.assertGreater(injection.head[-1].weight.grad.abs().sum().item(), 0)
            q = torch.randn(1, injection.project.out_channels, 4, 4)
            channels = injection.head[0].in_channels - q.shape[1]
            condition, time = torch.randn(1, channels, 4, 4), torch.randn(1, 512)
            with torch.no_grad():
                injection.head[-1].weight.add_(.1 * torch.randn_like(injection.head[-1].weight))
            weights = injection.weights(q, condition, time)
            self.assertTrue(torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0])))
            self.assertFalse(torch.allclose(weights, injection.weights(q + 1, condition, time)))
        self.assertIsNone(feedback._innovation_bank)
        self.assertIsNone(copy.deepcopy(feedback)._innovation_bank)

    def test_haar_parseval_mask_and_error_identity(self):
        x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
        mask = torch.ones_like(x, dtype=torch.bool)
        low, detail, valid = haar_step(x, mask)
        self.assertAlmostEqual(x.square().sum().item(), (low.square().sum() + detail.square().sum()).item())
        mask[..., :8, :8] = False
        y = torch.rand_like(x)
        p = y.clone()
        p[~mask] = 1000
        row = scene_diagnostics(p, y, mask)
        self.assertEqual(row["fine_mse"], 0.)
        self.assertEqual(row["db_rmse"], 0.)

    def test_gate_no_test_leakage_missing_or_failed_evidence(self):
        def evidence(fine):
            return dict(split="val", array_size="8x8", records=[dict(scene_id=f"u{s}", steps=n,
                realization=r, fine_mse=fine, db_rmse=10., db_bias=0.)
                for n in (2, 8) for r in range(4) for s in range(80)])
        results = {name: evidence(fine) for name, fine in zip(("B2", "M1", "M2", "M3"), (100, 98, 90, 80))}
        self.assertTrue(validation_gate(results)["launch_16x16_m3"])
        results["M3"] = evidence(95)
        self.assertFalse(validation_gate(results)["launch_16x16_m3"])
        results["M3"]["split"] = "test"
        with self.assertRaises(ValueError):
            validation_gate(results)


if __name__ == "__main__":
    unittest.main()
