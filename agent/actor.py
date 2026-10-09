"""PPO actor network and action-distribution utilities.

The actor represents the policy :math:`pi_theta(a | s)`. For image
observations it uses a convolutional encoder; for vector observations it uses
an MLP encoder. The policy head is categorical for discrete action spaces and
factorized diagonal Gaussian for continuous action spaces.

The methods in this module provide the exact quantities needed by PPO:

* action samples from :math:`pi_theta`,
* :math:`log pi_theta(a | s)` for the likelihood ratio,
* policy entropy for the entropy bonus, and
* deterministic actions for evaluation.

No environment-specific logic is included here.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


class ActorNetwork(nn.Module):
    """Policy network producing a distribution over environment actions.

    The network automatically selects an image encoder when the observation
    has at least two spatial dimensions. Image inputs are expected in either
    ``[batch, channels, height, width]`` or ``[batch, height, width]`` form.
    Vector observations are expected in ``[batch, features]`` form.

    Args:
        observation_shape: Shape of one observation, excluding batch size.
        action_dim: Number of discrete actions or continuous action dimensions.
        discrete_actions: Select a categorical policy when true; otherwise
            select a diagonal Gaussian policy.
        hidden_dim: Width of the vector policy trunk.
        image_input_scale: Divisor applied to image observations. The default
            converts uint8 pixel values from ``[0, 255]`` to approximately
            ``[0, 1]`` while leaving already normalized inputs unchanged.
        continuous_log_std_init: Initial log standard deviation for continuous
            actions. It is a trainable parameter, as in diagonal-Gaussian PPO.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_dim: int,
        discrete_actions: bool = True,
        hidden_dim: int = 512,
        image_input_scale: float = 255.0,
        continuous_log_std_init: float = 0.0,
    ) -> None:
        """Initialize the actor architecture and orthogonally initialize it.

        Input: Observation dimensions, action-space dimensions, action-space
            type, and architecture hyperparameters.
        Output: An initialized ``ActorNetwork`` module.
        Mathematical meaning: The parameters define a differentiable mapping
            from states to the policy distribution :math:`pi_theta(. | s)`.
        """
        super().__init__()
        if not observation_shape or any(d <= 0 for d in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("action_dim and hidden_dim must be positive")
        if image_input_scale <= 0.0:
            raise ValueError("image_input_scale must be positive")
        if len(observation_shape) == 1:
            self._is_image = False
            self.encoder = nn.Sequential(
                nn.Linear(observation_shape[0], hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            encoded_dim = hidden_dim
        elif len(observation_shape) in (2, 3):
            self._is_image = True
            channels, height, width = self._image_dimensions(observation_shape)
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                encoded_dim = int(
                    self.encoder(torch.zeros(1, channels, height, width)).shape[-1]
                )
            self.encoder.add_module("projection", nn.Sequential(
                nn.Linear(encoded_dim, hidden_dim),
                nn.ReLU(),
            ))
            encoded_dim = hidden_dim
        else:
            raise ValueError("observation_shape must be a vector, matrix, or image shape")

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.image_input_scale = image_input_scale
        self.policy_head = nn.Linear(encoded_dim, action_dim)
        if discrete_actions:
            self.log_std = None
        else:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), float(continuous_log_std_init))
            )
        self._initialize_weights()

    @staticmethod
    def _image_dimensions(observation_shape: tuple[int, ...]) -> Tuple[int, int, int]:
        """Convert a two- or three-dimensional observation shape to CHW.

        Input: ``observation_shape`` with either ``(height, width)`` or
            ``(channels, height, width)`` layout.
        Output: A ``(channels, height, width)`` tuple for convolutional layers.
        Mathematical meaning: This identifies the coordinate domain on which
            the visual feature extractor computes learned spatial features.
        """
        if len(observation_shape) == 2:
            return 1, observation_shape[0], observation_shape[1]
        first, second, third = observation_shape
        # Accept both conventional CHW tensors and environment-native HWC
        # tensors; Atari observations are commonly declared as (H, W, C).
        if first <= 4 and second > 4 and third > 4:
            return first, second, third
        return third, first, second

    def _initialize_weights(self) -> None:
        """Apply PPO-standard orthogonal initialization to trainable layers.

        Input: The modules owned by this actor.
        Output: No value; parameters are initialized in place.
        Mathematical meaning: Orthogonal initialization preserves signal scale
            through the initial policy mapping; the small policy-head gain
            starts the policy near a high-entropy, nearly uniform distribution.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def _prepare_observation(self, observation: Tensor) -> Tensor:
        """Validate, batch, and numerically prepare an observation tensor.

        Input: A tensor containing one observation or a batch of observations.
        Output: A floating-point tensor with a batch dimension and, for images,
            channel-first layout.
        Mathematical meaning: This computes the state representation supplied
            to the policy function without changing the represented state
            except for standard pixel scaling.
        """
        if observation.ndim == len(self.observation_shape):
            observation = observation.unsqueeze(0)
        if observation.ndim != len(self.observation_shape) + 1:
            raise ValueError("observation has an invalid number of dimensions")
        observation = observation.float()
        if self._is_image:
            if len(self.observation_shape) == 2:
                observation = observation.unsqueeze(1)
            elif self.observation_shape[0] <= 4 and self.observation_shape[1] > 4:
                if observation.shape[1] != self.observation_shape[0]:
                    raise ValueError("CHW image observations have the wrong channel dimension")
            else:
                if observation.shape[-1] != self.observation_shape[-1]:
                    raise ValueError("HWC image observations have the wrong channel dimension")
                observation = observation.permute(0, 3, 1, 2).contiguous()
            if observation.max().detach().item() > 1.0:
                observation = observation / self.image_input_scale
        elif observation.shape[-1] != self.observation_shape[0]:
            raise ValueError("vector observations have the wrong feature dimension")
        return observation

    def _distribution(self, observation: Tensor) -> Categorical | Normal:
        """Construct the policy distribution for a batch of observations.

        Input: A tensor containing one or more observations.
        Output: A ``Categorical`` distribution for discrete actions or a
            ``Normal`` distribution for continuous actions.
        Mathematical meaning: Computes :math:`pi_theta(. | s)` from the actor
            logits or Gaussian location and scale parameters.
        """
        prepared = self._prepare_observation(observation)
        features = self.encoder(prepared)
        output = self.policy_head(features)
        if self.discrete_actions:
            return Categorical(logits=output)
        assert self.log_std is not None
        scale = self.log_std.exp().expand_as(output)
        return Normal(output, scale)

    def forward(self, observation: Tensor) -> Tensor:
        """Return policy logits or means for the supplied observations.

        Input: One observation or a batch of observations.
        Output: Discrete-action logits of shape ``[batch, action_dim]`` or
            continuous-action means of the same shape.
        Mathematical meaning: Returns the parameterization of
            :math:`pi_theta(. | s)` before distribution construction.
        """
        prepared = self._prepare_observation(observation)
        return self.policy_head(self.encoder(prepared))

    def get_action_and_stats(
        self, observation: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample or select actions and calculate their log-probabilities.

        Input: One observation or a batch and a deterministic-action flag.
        Output: ``(action, log_probability, entropy)``. The action has shape
            ``[batch]`` for discrete policies and ``[batch, action_dim]`` for
            continuous policies; the two statistics have shape ``[batch]``.
        Mathematical meaning: Produces :math:`a_t`,
            :math:`log pi_theta(a_t|s_t)`, and
            :math:`H[pi_theta(.|s_t)]`, the quantities stored in a rollout.
        """
        distribution = self._distribution(observation)
        if deterministic:
            action = distribution.probs.argmax(dim=-1) if self.discrete_actions else distribution.mean
        else:
            action = distribution.sample()
        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()
        if not self.discrete_actions:
            log_probability = log_probability.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        return action, log_probability, entropy

    def evaluate_actions(
        self, observation: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evaluate stored actions under the current policy parameters.

        Input: A batch of observations and corresponding stored actions.
        Output: ``(log_probability, entropy)`` with one scalar per batch item.
        Mathematical meaning: Recomputes the numerator of
            :math:`r_t(theta)=exp(log pi_theta(a_t|s_t)-log pi_old(a_t|s_t))`
            and the entropy term used in the PPO loss.
        """
        distribution = self._distribution(observation)
        if self.discrete_actions:
            action = action.long().reshape(-1)
        else:
            action = action.float()
        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()
        if not self.discrete_actions:
            log_probability = log_probability.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        return log_probability, entropy

    def deterministic_action(self, observation: Tensor) -> Tensor:
        """Return the mode of the policy without sampling noise.

        Input: One observation or a batch of observations.
        Output: Greedy categorical actions or Gaussian means.
        Mathematical meaning: Computes the policy mode, suitable for
            evaluation episodes where stochastic exploration is disabled.
        """
        action, _, _ = self.get_action_and_stats(observation, deterministic=True)
        return action
