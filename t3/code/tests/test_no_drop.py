import json
from pathlib import Path
import unittest
import torch
from run_no_drop import NoDropUNO, NoDropConfig
from model.spectral_gate_uno import SpectralGateUNO2d
from training.matern_source import digest_state


class NoDropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_weights_and_rng_unchanged(self):
        torch.manual_seed(42)
        old = SpectralGateUNO2d('AP', operator_width=32)
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new = NoDropUNO(operator_width=32)
        self.assertEqual(old.cfg_drop_prob, .25)
        self.assertEqual(new.cfg_drop_prob, 0.)
        self.assertEqual(digest_state(old.state_dict()), digest_state(new.state_dict()))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_train_retains_raw_and_all_encoded_conditions(self):
        model = NoDropUNO(operator_width=32)
        # Spy only on the velocity call; use real encoder + shape validation.
        c, x, t = torch.randn(2, 3, 64, 64), torch.randn(2, 1, 64, 64), torch.tensor([.1, .7])
        embeds = model.embed_model(c)
        seen = []
        def spy(condition, state, time, embeddings):
            seen.append((condition, embeddings))
            return state
        model._velocity = spy
        model.train()
        for seed in range(4):
            torch.manual_seed(seed)
            model(image=c, x=x, step=t, embedding=embeds)
            self.assertTrue(torch.equal(seen[-1][0], c))
            for a, b in zip(seen[-1][1], embeds):
                self.assertTrue(torch.equal(a, b))
        seen.clear()
        model.forward_with_cfg(image=c, x=x, step=t, embedding=embeds, cfg_scale=1.)
        self.assertEqual(len(seen), 1)

    def test_full_velocity_eval_equivalence(self):
        torch.manual_seed(42)
        old = SpectralGateUNO2d('AP', operator_width=32).eval()
        torch.manual_seed(42)
        new = NoDropUNO(operator_width=32).eval()
        # Test the registered grid. The legacy AP gate is not valid at every
        # arbitrary small grid because retained-mode budgets are fixed.
        c, x, t = torch.randn(1,3,256,256), torch.randn(1,1,256,256), torch.tensor([.25])
        with torch.no_grad():
            a = old.forward_with_cfg(image=c, x=x, step=t, cfg_scale=1.)
            b = new(image=c, x=x, step=t)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
