import unittest
import torch
from model.spectral_gate_uno import SpectralGateUNO2d, SpectralGate, retained
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d


class SpectralGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_common_weights_rng_and_identity_stage(self):
        torch.manual_seed(42)
        original=AttentionMultiscaleUNO2d(operator_width=32)
        rng=torch.get_rng_state().clone()
        x=torch.randn(2,32,24,24);c=torch.randn(2,32,24,24);t=torch.randn(2,512)
        for variant in ('B','R','A','AP'):
            torch.manual_seed(42)
            model=SpectralGateUNO2d(variant,operator_width=32)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            for k,v in original.state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
            torch.testing.assert_close(model.encoder_stages[0](x,c,t),original.encoder_stages[0](x,c,t),rtol=0,atol=0)

    def test_gate_bounds_real_increment_and_gradients(self):
        for shape in ((24,24),(25,27)):
            for variant in ('R','A','AP'):
                torch.manual_seed(13)
                gate=SpectralGate(variant,3,4)
                with torch.no_grad():
                    if variant=='R':gate.radial_raw.fill_(.3)
                    else:
                        gate.amplitude_head.weight.normal_(0,.02)
                        if variant=='AP':gate.phase_head.weight.normal_(0,.02)
                x=torch.randn(2,4,*shape,requires_grad=True)
                c=torch.randn(2,3,*shape,requires_grad=True)
                h=retained(torch.fft.rfft2(x),4,4)
                g=gate(h,c,torch.randn(2,512),shape,(4,4),0)
                self.assertTrue((g.abs()>=.5-1e-6).all() and (g.abs()<=1.5+1e-6).all())
                torch.testing.assert_close(g[...,0],torch.ones_like(g[...,0]),rtol=0,atol=0)
                delta=torch.zeros(2,4,shape[0],shape[1]//2+1,dtype=torch.cfloat)
                delta[...,:4,:4]=(g[...,:4,:]-1)*h[...,:4,:]
                delta[...,-4:,:4]=(g[...,4:,:]-1)*h[...,4:,:]
                torch.testing.assert_close(torch.fft.rfft2(torch.fft.irfft2(delta,s=shape)),delta,rtol=1e-4,atol=1e-4)
                loss=(g.real.square()+.7*g.imag).mean();loss.backward()
                for name,p in gate.named_parameters():
                    self.assertIsNotNone(p.grad,name)
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                if variant!='R':
                    self.assertGreater(c.grad.abs().sum().item(),0)
                    other=gate(h.detach(),torch.zeros_like(c),torch.randn(2,512),shape,(4,4),0)
                    self.assertFalse(torch.equal(g.detach(),other))

    def test_zero_signal_is_finite_and_heads_learn_at_identity(self):
        for variant in ('A','AP'):
            gate=SpectralGate(variant,3,4)
            x=torch.zeros(2,4,8,8,requires_grad=True)
            c=torch.zeros(2,3,8,8,requires_grad=True)
            g=gate(retained(torch.fft.rfft2(x),2,3),c,torch.zeros(2,512),(8,8),(2,3),0)
            (g.real.sum()+g.imag.sum()).backward()
            self.assertTrue(torch.isfinite(x.grad).all() and torch.isfinite(c.grad).all())
            self.assertGreater(gate.amplitude_head.bias.grad.abs().sum().item(),0)
            if variant=='AP':self.assertGreater(gate.phase_head.bias.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
