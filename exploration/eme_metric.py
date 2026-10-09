"""EME's learned metric d_phi (Eq. 9) on top of the BiGAN embedding.

EME does not use a raw encoder distance as its state discrepancy. It learns a
pseudo-metric ``d_phi`` by regressing onto a target built from three terms
(Wang et al., 2024, Eq. 6 and 9): the *value-difference* term corrected by the
ensemble variance, a *bootstrapped next-state distance*, and a *policy-KL*
term. The KL term is what protects against the "Noisy-TV" failure mode: a
transition whose outcome is unpredictable but action-irrelevant (screen
flicker, death flashes) has high reward uncertainty but similar policy
distributions, so the learned metric stops rewarding it.

.. math::
    L(phi) = E[ ( d_phi(f_i, f_j)
        - sqrt(relu(|r_i - r_j|^2 - zeta(r_i) - zeta(r_j)))
        - gamma * stopgrad(d_phi(f'_i, f'_j))
        - gamma * KL(pi(.|s_i) || pi(.|s_j)) )^2 ]

where ``f = E(s)`` is the (frozen) BiGAN latent code. When trained, the
exploration bonus uses ``d_E(s_t, s_{t+1}) = d_phi(z_t, z_{t+1})`` as the
distance, inheriting EME's value-difference bound (Theorem 2) instead of the
untrained geometric displacement.

The encoder must be stationary while this metric is learned; the trainer
freezes it as soon as metric learning is enabled.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor, nn

from exploration.state_discrepancy import LatentStateDiscrepancy


class EMEMetricHead(nn.Module):
    """Non-negative regression head ``d_phi(z_i, z_j) -> R+``."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        """Build the pairwise metric MLP.

        Input: Latent dimension of the encoder codes and hidden width.
        Output: Initialized metric head.
        Mathematical meaning: Defines ``d_phi: Z x Z -> R+`` with softplus
            output non-negativity so the head is a valid discrepancy.
        """
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        # Scale the final layer down so the initial metric sits near zero and
        # the regression, not the initialization, sets the magnitude.
        with torch.no_grad():
            self.network[-2].weight.mul_(0.01)

    def forward(self, codes_i: Tensor, codes_j: Tensor) -> Tensor:
        """Return one distance per state pair.

        Input: Latent code batches ``[B, latent_dim]``.
        Output: Non-negative distances ``[B]``.
        Mathematical meaning: Computes ``d_phi(z_i, z_j)``.
        """
        paired = torch.cat([codes_i, codes_j], dim=-1)
        return self.network(paired).squeeze(-1)


class EMEMetricLearner:
    """Train ``d_phi`` with EME's Eq. (9) regression on rollout transitions.

    Args:
        discrepancy: The latent metric operator supplying the (frozen)
            encoder and the latent normalization used both for the regression
            inputs and for the bonus computation.
        actor: The policy network, used to recompute ``pi(.|s)`` for the KL
            term. Discrete policies use the exact categorical KL; continuous
            policies currently contribute a zero KL term (documented
            simplification).
        action_dim: Width of the action encoding (used only to validate the
            ensemble inputs).
        discrete_actions: Whether the action space is discrete.
        gamma: Discount factor weighting the bootstrapped next-state distance
            and the policy-KL term, exactly as in Eq. (4).
        hidden_dim: Hidden width of the metric head.
        learning_rate: Adam learning rate of the metric head.
        batch_size: Number of state pairs sampled per update.
        max_grad_norm: Gradient-norm clip of the metric head.
        device: Torch device for the head and its optimizer.
        generator: Optional seeded generator for pair sampling.
    """

    def __init__(
        self,
        discrepancy: LatentStateDiscrepancy,
        actor: nn.Module,
        action_dim: int,
        discrete_actions: bool,
        gamma: float,
        hidden_dim: int = 256,
        learning_rate: float = 1.0e-3,
        batch_size: int = 256,
        max_grad_norm: float = 0.5,
        device: Union[torch.device, str] = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Create the head, optimizer, and sampling state.

        Input: Latent discrepancy operator, policy network, action-space
            description, discount, architecture, and optimization settings.
        Output: Initialized metric learner.
        Mathematical meaning: Instantiates the regression problem
            ``min_phi L(phi)`` of Eq. (9) over rollout replay pairs.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        self.discrepancy = discrepancy
        self.actor = actor
        self.action_dim = int(action_dim)
        self.discrete_actions = bool(discrete_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device(device)
        self.generator = generator
        self.head = EMEMetricHead(discrepancy.latent_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=learning_rate)
        self.update_count = 0

    def _pair_index_generator(self, total: int, count: int) -> Tensor:
        """Sample ``count`` random transition indices in ``[0, total)``.

        Input: Population size and sample count.
        Output: Long tensor of sampled indices.
        Mathematical meaning: Draws the pair support of one minibatch of the
            Eq. (9) expectation.
        """
        return torch.randint(
            0, total, (count,), generator=self.generator, device=self.device
        )

    @torch.no_grad()
    def _policy_kl(self, observations_i: Tensor, observations_j: Tensor) -> Tensor:
        """Return ``KL(pi(.|s_i) || pi(.|s_j))`` for each sampled pair.

        Input: Observation batches for both sides of every pair.
        Output: Non-negative tensor ``[B]``.
        Mathematical meaning: Evaluates the behavioral-similarity term of
            Eq. (4) under the current policy; for discrete actions this is the
            exact categorical KL, for continuous actions the term is currently
            approximated by zero.
        """
        if not self.discrete_actions:
            return torch.zeros(observations_i.shape[0], device=self.device)
        logits_i = self.actor(observations_i.to(self.device).float())
        logits_j = self.actor(observations_j.to(self.device).float())
        distribution_i = torch.distributions.Categorical(logits=logits_i)
        distribution_j = torch.distributions.Categorical(logits=logits_j)
        return torch.distributions.kl.kl_divergence(distribution_i, distribution_j)

    def update(
        self,
        observations: Tensor,
        actions: Tensor,
        extrinsic_rewards: Tensor,
        terminated: Tensor,
        ensemble=None,
    ) -> float:
        """Run one Eq. (9) regression step on sampled transition pairs.

        Input: Time-major rollout tensors -- observations ``[T+1, N, ...]``
            (the rollout-buffer layout whose slot ``t+1`` is the successor of
            slot ``t``), actions ``[T, N]``, extrinsic rewards ``[T, N]``,
            termination mask ``[T, N]``, and the reward ensemble used for the
            variance correction (optional; zeros when absent).
        Output: The regression loss of this step as a float.
        Mathematical meaning: Minimizes one stochastic estimate of ``L(phi)``
            with fresh random pairs ``(s_i, s_j)`` from the rollout.
        """
        steps, num_envs = extrinsic_rewards.shape
        total = steps * num_envs
        observations = observations.to(self.device).float()
        flat_observations = observations.reshape(-1, *observations.shape[2:])
        # Slot t+1 of the buffer is the successor state of slot t.
        rewards = extrinsic_rewards.detach().reshape(-1).to(self.device).float()
        terminated_mask = terminated.detach().reshape(-1).to(self.device).bool()

        index_i = self._pair_index_generator(total, self.batch_size)
        index_j = self._pair_index_generator(total, self.batch_size)
        successor_index_i = index_i + num_envs
        successor_index_j = index_j + num_envs

        with torch.no_grad():
            codes_i = self.discrepancy.normalize_codes(
                self.discrepancy.encode(flat_observations[index_i])
            )
            codes_j = self.discrepancy.normalize_codes(
                self.discrepancy.encode(flat_observations[index_j])
            )
            codes_successor_i = self.discrepancy.normalize_codes(
                self.discrepancy.encode(flat_observations[successor_index_i])
            )
            codes_successor_j = self.discrepancy.normalize_codes(
                self.discrepancy.encode(flat_observations[successor_index_j])
            )
            if ensemble is not None:
                flat_actions = actions.detach().reshape(-1).to(self.device)
                zeta_i = ensemble.get_variance(
                    flat_observations[index_i], flat_actions[index_i]
                ).float()
                zeta_j = ensemble.get_variance(
                    flat_observations[index_j], flat_actions[index_j]
                ).float()
            else:
                zeta_i = torch.zeros(self.batch_size, device=self.device)
                zeta_j = torch.zeros(self.batch_size, device=self.device)
            reward_gap_squared = (rewards[index_i] - rewards[index_j]) ** 2
            # Unbiased correction of EME Proposition 3: subtract the ensemble
            # variances before taking the square root; a negative remainder
            # means the pair carries no resolvable value difference.
            value_difference = torch.sqrt(
                torch.relu(reward_gap_squared - zeta_i - zeta_j)
            )
            bootstrapped = self.head(codes_successor_i, codes_successor_j)
            policy_kl = self._policy_kl(
                flat_observations[index_i], flat_observations[index_j]
            )
            terminal = terminated_mask[index_i] | terminated_mask[index_j]
            # The bootstrapped term is undefined through a terminal state.
            bootstrapped = bootstrapped * (~terminal).float()

        target = (
            value_difference
            + self.gamma * bootstrapped
            + self.gamma * policy_kl
        )
        prediction = self.head(codes_i, codes_j)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.update_count += 1
        return float(loss.detach().cpu())
