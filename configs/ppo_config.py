from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 2048
    hidden_dims: list = None

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
