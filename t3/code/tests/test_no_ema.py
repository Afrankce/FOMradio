import copy
import unittest
import torch
from training.ema import ModelEMA
from training.no_ema_experiment import OnlineWeights


class NoEMATests(unittest.TestCase):
    def test_no_average_and_same_optimizer_trajectory(self):
        torch.manual_seed(42)
        a=torch.nn.Linear(3,2); b=copy.deepcopy(a)
        ema=ModelEMA(a); online=OnlineWeights(b)
        oa=torch.optim.AdamW(a.parameters(),lr=.01)
        ob=torch.optim.AdamW(b.parameters(),lr=.01)
        for _ in range(4):
            x=torch.randn(5,3);y=torch.randn(5,2)
            for model,optimizer,weights in ((a,oa,ema),(b,ob,online)):
                optimizer.zero_grad();(model(x)-y).square().mean().backward()
                optimizer.step();weights.update(model)
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertIs(online.ema_model,b)
        self.assertTrue(all(p.requires_grad for p in b.parameters()))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(a.parameters(),ema.ema_model.parameters())))

    def test_no_rng_consumption(self):
        model=torch.nn.Linear(3,2);state=torch.get_rng_state()
        weights=OnlineWeights(model);weights.update(model)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
