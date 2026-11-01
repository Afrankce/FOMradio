import unittest
import torch
from training.detail_losses import detail_components, auxiliary, gradient_statistics


class DetailLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.x = torch.randn(3,1,32,32,requires_grad=True)
        self.y = torch.randn_like(self.x)
        self.mask = torch.ones_like(self.x,dtype=torch.bool)
        self.mask[...,10:14,8:12] = False

    def test_mask_invariance_and_gradient(self):
        a = detail_components(self.x,self.y,self.mask)
        changed = self.y.clone(); changed[~self.mask] = float('nan')
        b = detail_components(self.x,changed,self.mask)
        for k in a:
            torch.testing.assert_close(a[k][0],b[k][0])
        means = {k:s/n for k,(s,n) in b.items()}
        for v in ('L1','L2'):
            g = torch.autograd.grad(means['fm']+.1*auxiliary(means,v),self.x,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(g).all())
            self.assertEqual(float(g[~self.mask].abs().sum()),0.)

    def test_identity_and_constant(self):
        for a,b in ((self.x,self.x),(self.x,self.x+2)):
            terms = detail_components(a,b,self.mask)
            for k in ('gradient','haar1','haar2'):
                self.assertLess(float(terms[k][0]),1e-8)

    def test_partition_global_normalization(self):
        whole=detail_components(self.x,self.y,self.mask)
        pieces=[detail_components(self.x[i:i+1],self.y[i:i+1],self.mask[i:i+1]) for i in range(3)]
        for k in whole:
            torch.testing.assert_close(whole[k][0],sum(p[k][0] for p in pieces))
            self.assertEqual(int(whole[k][1]),sum(int(p[k][1]) for p in pieces))

    def test_objective_gradient_linearity(self):
        terms=detail_components(self.x,self.y,self.mask)
        means={k:s/n for k,(s,n) in terms.items()}
        a,b=means['fm'],auxiliary(means,'L2')
        ga,gb=[torch.autograd.grad(z,self.x,retain_graph=True)[0] for z in (a,b)]
        total=torch.autograd.grad(a+.1*b,self.x)[0]
        torch.testing.assert_close(total,ga+.1*gb)
        stat=gradient_statistics(ga.flatten(),gb.flatten(),.1)
        self.assertTrue(-1.00001<=stat['cosine']<=1.00001)


if __name__=='__main__': unittest.main()
