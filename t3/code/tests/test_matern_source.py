import unittest
import numpy as np
from scipy.special import k1
import torch
from model.matern_source import SourceSampler,source_spectra,path_triplet,SOURCES
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from training.matern_source import digest_state


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_covariance_and_mix(self):
        n,e=source_spectra()
        self.assertGreaterEqual(e.min(),0)
        np.testing.assert_allclose(e[4],e[1:4].mean(0),rtol=1e-12)
        for index,length in enumerate((2,8,32),1):
            c=np.fft.irfft2(e[index],s=(n,n))
            self.assertAlmostEqual(c[0,0],1.,places=10)
            for r in (1,4,16,64): self.assertAlmostEqual(c[0,r],r/length*k1(r/length),places=9)

    def test_rng_common_base_and_repeat(self):
        meta=[dict(scene_id='u10',steering_deg=0)]
        state=torch.get_rng_state().clone();a=SourceSampler('W');b=SourceSampler('MS')
        wa=a.base_noise(meta,'cpu','test');wb=b.base_noise(meta,'cpu','test')
        self.assertTrue(torch.equal(wa,wb));self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(b.sample(meta,device='cpu',namespace='test'),b.sample(meta,device='cpu',namespace='test')))
        self.assertTrue(torch.equal(a.transform(wa),wa[...,:256,:256]))

    def test_monte_carlo(self):
        # 32 independent realizations; pool pixels but use broad MC sanity bounds.
        meta=[dict(scene_id=f'u{i}',steering_deg=0) for i in range(32)]
        for source in SOURCES:
            sampler=SourceSampler(source)
            x=sampler.sample(meta,device='cpu',namespace='mc')
            self.assertLess(abs(float(x.mean())),.15)
            self.assertLess(abs(float(x.square().mean())-1),.25)
            target=0 if source=='W' else np.mean([1/l*k1(1/l) for l in (2,8,32)]) if source=='MS' else 1/float(source[1:])*k1(1/float(source[1:]))
            observed=float((x[...,:-1]*x[...,1:]).mean())
            self.assertLess(abs(observed-target),.25)

    def test_path(self):
        a=torch.randn(2,1,8,8);b=torch.randn_like(a)
        for t in (0.,.3,1.):
            xt,ut=path_triplet(a,b,torch.full((2,),t))
            torch.testing.assert_close(ut,b-a)
            torch.testing.assert_close(xt,(1-t)*a+t*b)
        dt=.001;lo,_=path_triplet(a,b,torch.full((2,),.3-dt));hi,_=path_triplet(a,b,torch.full((2,),.3+dt))
        torch.testing.assert_close((hi-lo)/(2*dt),b-a,atol=.0003,rtol=.0003)

    def test_identical_architecture(self):
        hashes=[]
        for source in SOURCES:
            torch.manual_seed(42);m=AttentionMultiscaleUNO2d(operator_width=32)
            self.assertFalse(any('injection' in k for k in m.state_dict()))
            hashes.append(digest_state(m.state_dict()))
        self.assertEqual(len(set(hashes)),1)


if __name__=='__main__': unittest.main()
