import unittest
import numpy as np
import torch
from beam_matern_source import spatial_gradient, probability_map, variance_from_amplitudes, filter_noise


class SourceTests(unittest.TestCase):
    def test_uniform_has_unit_variance_and_differentiable_score(self):
        n=16; b=torch.ones(1,1,n,n);alpha=torch.tensor(1.,requires_grad=True);beta=torch.tensor(1.,requires_grad=True)
        q=probability_map(b,alpha,beta,1.)
        self.assertTrue(torch.allclose(q,torch.full_like(q,1/(n*n)),atol=1e-8))
        self.assertTrue(torch.allclose(variance_from_amplitudes(torch.ones(1,1,n,n),torch.ones(3,n,n//2+1)),torch.ones(1,1,n,n),atol=1e-6))

    def test_fft_variance_matches_direct_filter_matrix(self):
        torch.manual_seed(7);n=8
        kernels=torch.randn(3,n,n)
        amplitudes=torch.fft.rfft2(kernels).abs()
        kernel=torch.fft.irfft2(amplitudes,s=(n,n))
        rho=torch.rand(1,1,n,n)+.1
        actual=variance_from_amplitudes(rho,amplitudes)
        expected=torch.zeros_like(actual)
        for i in range(n):
            for j in range(n):
                for k in range(3):
                    a=(i-torch.arange(n))%n; b=(j-torch.arange(n))%n
                    expected[0,0,i,j] += (rho[0,0] * kernel[k].square()[a[:,None],b[None,:]]).sum()/3
        self.assertTrue(torch.allclose(actual,expected,rtol=1e-5,atol=1e-5), (actual-expected).abs().max())

    def test_filtered_white_uses_declared_operator(self):
        torch.manual_seed(7); n=8
        white=torch.randn(2,1,n,n);a=torch.ones(n,n//2+1)
        self.assertTrue(torch.allclose(filter_noise(white,a),white,atol=1e-6))

    def test_variance_matches_monte_carlo(self):
        torch.manual_seed(21);n=8; draws=8192
        kernel=torch.randn(n,n);amplitude=torch.fft.rfft2(kernel).abs()
        rho=torch.rand(1,1,n,n)+.2
        theory=variance_from_amplitudes(rho,amplitude[None].expand(3,-1,-1))
        white=torch.randn(draws,1,n,n)
        field=filter_noise(white*rho.sqrt(),amplitude)
        empirical=field.var(dim=0,unbiased=True,keepdim=True)
        self.assertLess(float(((empirical-theory).abs()/theory).max()),.06)


if __name__ == '__main__': unittest.main()
