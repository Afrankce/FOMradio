import unittest
import torch
from model.conditional_operator_uno import ConditionalOperatorUNO2d, ConditionModulation, ConditionalStage
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from training.matern_source import digest_state


class ConditionalOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_weights_and_budget(self):
        torch.manual_seed(42)
        original = AttentionMultiscaleUNO2d(operator_width=32).state_dict()
        models = {}
        for variant in ('CASA','ADD','FILM','NSOP'):
            torch.manual_seed(42)
            model = ConditionalOperatorUNO2d(variant, operator_width=32)
            models[variant] = model
            common = set(original) & set(model.state_dict())
            for key in common:
                self.assertTrue(torch.equal(original[key],model.state_dict()[key]), key)
            if variant == 'CASA':
                self.assertEqual(digest_state(original),digest_state(model.state_dict()))
            else:
                self.assertFalse(any('channel_attention' in k or 'spatial_attention' in k
                                     for k in model.state_dict()))
        self.assertEqual(sum(p.numel() for p in models['FILM'].parameters()),
                         sum(p.numel() for p in models['NSOP'].parameters()))
        self.assertEqual(digest_state(models['FILM'].state_dict()),digest_state(models['NSOP'].state_dict()))

    def test_stage_shapes_gradients_condition_dependency(self):
        for variant in ('FILM','NSOP'):
            torch.manual_seed(42)
            model=ConditionalOperatorUNO2d(variant,operator_width=32)
            stage=model.encoder_stages[0]
            x=torch.randn(2,32,24,24,requires_grad=True)
            c=torch.randn(2,stage._embedding_channels,24,24,requires_grad=True)
            t=torch.randn(2,512)
            y=stage(x,c,t)
            self.assertEqual(y.shape,x.shape)
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(c.grad.abs().sum().item(),0)
            for p in stage.condition_operator.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(),0)
            self.assertFalse(torch.equal(y.detach(),stage(x.detach(),torch.zeros_like(c),t)))

    def test_zero_modulation_recovers_add_stage(self):
        torch.manual_seed(42)
        base=ConditionalOperatorUNO2d('ADD',operator_width=32).encoder_stages[0]
        for variant in ('FILM','NSOP'):
            torch.manual_seed(42)
            stage=ConditionalOperatorUNO2d(variant,operator_width=32).encoder_stages[0]
            with torch.no_grad():
                stage.condition_operator.net[-1].weight.zero_()
                stage.condition_operator.net[-1].bias.zero_()
            x=torch.randn(1,32,24,24);c=torch.randn(1,stage._embedding_channels,24,24);t=torch.randn(1,512)
            torch.testing.assert_close(stage(x,c,t),base(x,c,t),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
