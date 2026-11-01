import unittest
import torch
from cross_array.model import SourceObservationUNO
from cross_array.paths import initial_state, source_mean
from model.matern_source import path_triplet

CONFIG=dict(experiment_kind='cross_array_transport',source_mean_rule='fixed_real_y8_0deg_masked_zero',source_noise_scale=1.)


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_identical_velocity_network(self):
        torch.manual_seed(42);a=SourceObservationUNO('T2')
        torch.manual_seed(42);b=SourceObservationUNO('T3')
        for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        c=torch.rand(1,6,128,128);x=torch.randn(1,1,128,128)
        torch.testing.assert_close(a(image=c,x=x,step=.3),b(image=c,x=x,step=.3),rtol=0,atol=0)

    def test_source_mask_and_legacy_identity(self):
        c=torch.rand(2,6,8,8);c[:,5]=1;c[:,5,:2]=0;c[:,4,:2]=1000
        z=torch.randn(2,1,8,8);x=initial_state(c,z,CONFIG);mu=source_mean(c)
        torch.testing.assert_close(x[:,:,:2],z[:,:,:2],rtol=0,atol=0)
        torch.testing.assert_close(x,z+mu,rtol=0,atol=0)
        self.assertIs(initial_state(c,z,dict(experiment_kind='cross_array_observation')),z)
        self.assertGreater(float((x-z).abs().max()),0)
        with self.assertRaises(ValueError):initial_state(c,z,{**CONFIG,'source_noise_scale':.5})

    def test_path_endpoints_derivative_and_residual_translation(self):
        c=torch.rand(2,6,8,8,dtype=torch.float64);c[:,5]=1
        z=torch.randn(2,1,8,8,dtype=torch.float64);y=torch.rand_like(z)
        mu=source_mean(c);x0=initial_state(c,z,CONFIG)
        for t in (0.,.35,1.):
            times=torch.full((2,),t,dtype=torch.float64);xt,u=path_triplet(x0,y,times)
            rt,ur=path_triplet(z,y-mu,times)
            torch.testing.assert_close(xt-mu,rt,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(u,ur,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(xt,x0+t*u,rtol=1e-12,atol=1e-12)
        state=x0.clone()
        for _ in range(8):state=state+(y-x0)/8
        torch.testing.assert_close(state,y,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
