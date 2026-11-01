import unittest
import torch
from beam_boundary.model import BoundaryUNO, BoundaryResidual, amplitude_lift, BeamPhaseGate
from beam_boundary.common import phase_choice
from beam_boundary.train import indices_for


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def inputs(self):
        c = torch.zeros(1, 3, 64, 64); c[:, 0, 31, 31] = 1; c[:, 1, 20:40, 20:40] = .3
        c[:, 2] = torch.linspace(.4, .9, 64)[None, None, :]
        return c

    def test_initialization_and_parameter_match(self):
        torch.manual_seed(42); reference = BoundaryUNO('E0').state_dict(); counts = {}
        for arm in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'P-A', 'P-AP'):
            torch.manual_seed(42); model = BoundaryUNO(arm)
            for key, value in reference.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]), (arm, key))
            counts[arm] = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
        self.assertEqual(counts['E2'], counts['E4']); self.assertEqual(counts['P-A'], counts['P-AP'])

    def test_gate_ablation(self):
        c = self.inputs(); altered = c.clone(); altered[:, 2] = .8
        for arm in ('E1', 'E2', 'E3', 'E5'):
            torch.manual_seed(17); module = BoundaryResidual(arm)
            gate = module.source_weights(c, (32, 32))
            self.assertTrue(torch.isfinite(gate).all()); self.assertGreaterEqual(float(gate.min()), 0)
            self.assertLessEqual(float(gate.max()), 1)
            if arm == 'E1': self.assertTrue(torch.equal(gate, gate[..., :1, :1].expand_as(gate)))
            if arm == 'E3': self.assertTrue(torch.equal(gate, module.source_weights(altered, (32, 32))))
            else: self.assertFalse(torch.equal(gate, module.source_weights(altered, (32, 32))))

    def test_zero_residual_and_gradient_opening(self):
        for arm in ('E2', 'E4'):
            module = BoundaryResidual(arm); h = torch.randn(1, 128, 32, 32)
            embedding = torch.randn(1, 64, 32, 32); time = torch.randn(1, 512)
            value = module(h, embedding, time, self.inputs())
            self.assertEqual(float(value.abs().max()), 0)
            value.sum().backward(); self.assertGreater(float(module.output_projection.weight.grad.abs().max()), 0)
            with torch.no_grad(): module.output_projection.weight.normal_(std=.01)
            module.zero_grad(); module(h, embedding, time, self.inputs()).square().sum().backward()
            self.assertGreater(float(module.source_gate[0].weight.grad.abs().sum()), 0)

    def test_hermitian_column(self):
        weights = BoundaryResidual('E2').spectral_weights()
        self.assertTrue(torch.equal(weights[..., 0], weights[..., 0].flip(-1).conj()))

    def test_phase_boundary_and_initial_identity(self):
        gate = BeamPhaseGate(16, True); modes = 8; rows = torch.arange(-7, 8).remainder(48)
        h = torch.randn(2, 16, 15, 8, dtype=torch.cfloat); beam = torch.rand(2, 1, 32, 32)
        result = gate(h, beam, 8, rows, modes)
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        with torch.no_grad(): gate.rotation.bias.fill_(1); gate.amplitude.bias.fill_(1)
        result = gate(h, beam, 8, rows, modes)
        self.assertTrue(torch.equal(result[..., 0], torch.ones_like(result[..., 0])))

    def test_amplitude_finite_monotone(self):
        result = amplitude_lift(torch.linspace(0, 1, 1000))
        self.assertTrue(torch.isfinite(result).all()); self.assertTrue((result[1:] >= result[:-1]).all())

    def test_balanced_addressed_batch(self):
        records = [dict(scene_id=str(i), steering_deg=a) for i in range(560) for a in (-14., 0., 14.)]
        indices = indices_for(records, 19)
        self.assertEqual(indices, indices_for(records, 19)); self.assertEqual(len(set(indices)), 128)
        counts = [sum(records[i]['steering_deg'] == a for i in indices) for a in (-14., 0., 14.)]
        self.assertEqual(sorted(counts), [42, 43, 43])

    def test_validation_selection_gate(self):
        scores = {a: dict(map_rmse_db=10., unseen_delta_rmse_db=10.) for a in ('E0', 'E2', 'E3', 'E5')}
        self.assertIsNone(phase_choice(scores))
        scores['E2']['unseen_delta_rmse_db'] = 9.8; self.assertEqual(phase_choice(scores), 'E2')
        scores['E2']['map_rmse_db'] = 10.2; self.assertIsNone(phase_choice(scores))


if __name__ == '__main__': unittest.main()
