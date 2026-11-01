"""No weight averaging: evaluate and checkpoint the online optimizer weights."""
from dataclasses import dataclass
from training.conditional_operator_experiment import ConditionalConfig, ConditionalTrainer


class OnlineWeights:
    """Compatibility adapter, not an EMA or a teacher; stores no second model."""
    def __init__(self, model):
        self.ema_model = model

    def update(self, model):
        if model is not self.ema_model:
            raise RuntimeError('Online weight adapter model mismatch')


@dataclass(frozen=True)
class NoEMAConfig(ConditionalConfig):
    def scientific_payload(self):
        return {**super().scientific_payload(), 'experiment': 'no_ema_conditional_operator_v1',
                'ema_enabled': False, 'ema_decay': None,
                'evaluation_weights': 'online optimizer weights, no averaging',
                'checkpoint_selection': 'epoch1000_last_online',
                'compatibility_ema_slot': 'alias of model, not averaged; both slots must match'}


class NoEMATrainer(ConditionalTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent checkpoint/evaluation interface expects ema_model.
        # Replace the temporary copy with an alias; do not freeze online parameters.
        self.ema = OnlineWeights(self.model)

    def resume(self, path):
        import torch
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if any(not torch.equal(v, payload['ema'][k]) for k,v in payload['model'].items()):
            raise RuntimeError('No-EMA checkpoint has unequal compatibility weights')
        del payload
        return super().resume(path)
