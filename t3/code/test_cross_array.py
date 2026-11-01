import unittest
import numpy as np
import torch
from cross_array.model import SourceObservationUNO
from cross_array.prepare import normalize_map
from cross_array.geometry import regions_from_height, regional_metrics
from cross_array.analyze import paired, decide


class CrossArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_identical_parameters_and_source_ablation(self):
        torch.manual_seed(42); a=SourceObservationUNO('T0')
        torch.manual_seed(42); b=SourceObservationUNO('T2')
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        c=torch.rand(1,6,128,128); altered=c.clone();altered[:,4:6]=.1
        self.assertTrue(torch.equal(a.sanitize(c),a.sanitize(altered)))
        self.assertFalse(torch.equal(b.sanitize(c),b.sanitize(altered)))
        x=torch.randn(1,1,128,128)
        va=a(image=c,x=x,step=.5);vb=b(image=c,x=x,step=.5)
        torch.testing.assert_close(va,vb,rtol=0,atol=0)
        vb.square().mean().backward()
        self.assertGreater(float(b.lifting.weight.grad[:,5:7].abs().sum()),0)
        with torch.no_grad():
            b.lifting.weight[:,5:7].fill_(.01);a.load_state_dict(b.state_dict())
        torch.testing.assert_close(a(image=c,x=x,step=.5),a(image=altered,x=x,step=.5),rtol=0,atol=0)
        self.assertGreater(float((b(image=c,x=x,step=.5)-b(image=altered,x=x,step=.5)).abs().max()),0)

    def test_source_sentinels_and_mask(self):
        y=np.full((128,128),-90.,np.float32);y[:8,:8]=1000;y[-8:,-8:]=-300
        value,mask=normalize_map(y)
        self.assertTrue(torch.isfinite(value).all());self.assertGreaterEqual(float(value.min()),0)
        self.assertLessEqual(float(value.max()),1);self.assertTrue(torch.equal(value[~mask],torch.zeros_like(value[~mask])))
        with self.assertRaises(AssertionError):normalize_map(np.full((128,128),1001.))

    def test_geometry_proxy_and_gradient_support(self):
        coords=dict(resolution=5,min_x=-637.5,max_x=637.5,min_y=-637.5,max_y=637.5)
        height=np.zeros((256,256));clear=regions_from_height(height,coords)
        self.assertFalse(clear['shadow'].any());self.assertFalse(clear['transition'].any())
        height[110:146,150:152]=80.;r=regions_from_height(height,coords)
        self.assertTrue(r['shadow'][64,100]);self.assertFalse(r['shadow'][64,40]);self.assertTrue(r['transition'].any())
        valid=r['free'];truth=np.full((128,128),-90.)
        scores=regional_metrics(truth,truth,valid,r)
        self.assertEqual(scores['shadow_transition_gradient_db_per_m'],0.)
        changed=truth.copy();changed[~valid]=1e9
        self.assertEqual(regional_metrics(changed,truth,valid,r)['shadow_transition_gradient_db_per_m'],0.)

    def test_progression_gate_and_missing_support(self):
        # Synthetic fixtures test decision logic only, never populate run results.
        effect=paired({str(i):9. for i in range(80)},{str(i):10. for i in range(80)},'val','fixture')
        effects={k:dict(effect) for k in ('unseen_delta_rmse_db','native_near_rmse_db','shadow_transition_gradient_db_per_m')}
        scores={'T0':dict(map_rmse_db=10.,native_rmse_db=10.),'T2':dict(map_rmse_db=9.,native_rmse_db=9.)}
        gate=dict(unseen_delta_improvement=.01,native_near_improvement=.03,
            shadow_transition_gradient_improvement=.03,map_guardrail=1.01,min_region_scenes=20)
        self.assertEqual(decide(scores,effects,gate)['decision'],'supported')
        scores['T2']['map_rmse_db']=11.
        self.assertEqual(decide(scores,effects,gate)['decision'],'not_supported')
        effects['native_near_rmse_db']=paired({}, {},'val','empty')
        self.assertEqual(decide(scores,effects,gate)['decision'],'inconclusive')
        with self.assertRaises(RuntimeError):paired({'a':1.},{'b':1.},'val','mismatch')


if __name__=='__main__':unittest.main()
