"""Training schedule for the BiGAN encoder, generator, and discriminator.

This module performs adversarial optimization from a replay buffer of
observations. It intentionally does not know about PPO or environment stepping;
the top-level trainer decides when to add observations and invoke ``update``.
"""

from __future__ import annotations

from typing import Union

from dataclasses import dataclass

import torch
from torch import Tensor

from config import BiGANConfig
from bigan.discriminator import BiGANDiscriminator
from bigan.encoder import BiGANEncoder
from bigan.generator import BiGANGenerator
from bigan.losses import (
    discriminator_loss,
    encoder_generator_loss,
    pixel_reconstruction_loss,
)
from utils.replay import ReplayBuffer


@dataclass(frozen=True)
class BiGANUpdateMetrics:
    """Aggregate diagnostics from one scheduled BiGAN update."""

    updated: bool
    discriminator_loss: float
    encoder_generator_loss: float
    reconstruction_loss: float
    discriminator_real_logit: float
    discriminator_fake_logit: float
    update_count: int


class BiGANTrainer:
    """Optimize the three BiGAN networks from replayed observations.

    Args:
        encoder: Observation-to-latent encoder ``E_psi``.
        generator: Latent-to-observation generator ``G_theta``.
        discriminator: Joint discriminator ``D_omega``.
        config: Validated BiGAN hyperparameters.
        observation_shape: Shape of replay observations.
        device: Training device.
    """

    def __init__(
        self,
        encoder: BiGANEncoder,
        generator: BiGANGenerator,
        discriminator: BiGANDiscriminator,
        config: BiGANConfig,
        observation_shape: tuple[int, ...],
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize BiGAN networks, optimizers, and schedule counters.

        Input: Three compatible networks, BiGAN configuration, observation
            shape, and training device.
        Output: An initialized ``BiGANTrainer``.
        Mathematical meaning: Creates optimization processes for
            ``E_psi``, ``G_theta``, and ``D_omega`` in the bidirectional
            adversarial game.
        """
        if tuple(encoder.observation_shape) != tuple(observation_shape):
            raise ValueError("encoder and trainer observation shapes differ")
        if tuple(generator.observation_shape) != tuple(observation_shape):
            raise ValueError("generator and trainer observation shapes differ")
        if tuple(discriminator.observation_shape) != tuple(observation_shape):
            raise ValueError("discriminator and trainer observation shapes differ")
        if encoder.latent_dim != generator.latent_dim or encoder.latent_dim != discriminator.latent_dim:
            raise ValueError("encoder, generator, and discriminator latent dimensions differ")

        self.encoder = encoder.to(device)
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.observation_shape = observation_shape
        self.device = torch.device(device)
        self.update_count = 0
        self.encoder_frozen = False

        betas = (config.adam_beta1, config.adam_beta2)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_epsilon,
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_epsilon,
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_epsilon,
        )

    def _model_observations(self, observations: Tensor) -> Tensor:
        """Convert replay observations to the generator's numeric domain.

        Input: Observation batch in replay storage dtype/layout.
        Output: Floating-point observations, scaled to ``[0,1]`` when values
            are represented as raw pixel intensities above one.
        Mathematical meaning: Places real observations and generated outputs in
            the same domain for the pixel reconstruction term.
        """
        values = observations.to(self.device).float()
        if (
            len(self.observation_shape) in (2, 3)
            and values.numel() > 0
            and values.detach().amax().item() > 1.0
        ):
            values = values / 255.0
        return values

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
        """Enable or disable gradients for all parameters in a module.

        Input: A module and the desired ``requires_grad`` state.
        Output: No value; parameter flags are changed in place.
        Mathematical meaning: Prevents discriminator parameters from receiving
            encoder/generator update gradients while retaining differentiability
            of the discriminator output with respect to its inputs.
        """
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def freeze_encoder(self) -> None:
        """Stop optimizing the encoder while continuing adversarial training.

        Input: This trainer.
        Output: No value; encoder parameters stop receiving gradients and the
            encoder optimizer step is skipped from now on.
        Mathematical meaning: Fixes ``E_psi`` after pretraining so that any
            downstream latent metric, such as the Contribution 2 exploration
            bonus ``||E(s_t)-E(s_{t+1})||_p``, is measured in a stationary
            space instead of one that drifts with the adversarial game.
        """
        if self.encoder_frozen:
            return
        self._set_requires_grad(self.encoder, False)
        self.encoder.eval()
        self.encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        """Resume encoder optimization after a freeze.

        Input: This trainer.
        Output: No value; encoder parameters receive gradients again.
        Mathematical meaning: Restores joint optimization of ``E_psi`` in the
            bidirectional adversarial objective.
        """
        if not self.encoder_frozen:
            return
        self._set_requires_grad(self.encoder, True)
        self.encoder.train()
        self.encoder_frozen = False

    def _sample_latents(self, batch_size: int) -> Tensor:
        """Sample standard-normal latent vectors for generated joint pairs.

        Input: Positive latent batch size.
        Output: Tensor ``[batch_size, latent_dim]`` on the training device.
        Mathematical meaning: Samples ``z ~ N(0,I)`` from the BiGAN prior used
            in ``(G_theta(z), z)``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.randn(
            batch_size,
            self.encoder.latent_dim,
            device=self.device,
        )

    def _discriminator_step(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Perform one discriminator optimization step.

        Input: A floating-point replay batch of real observations.
        Output: ``(loss, mean_real_logit, mean_fake_logit)`` detached from the
            update graph.
        Mathematical meaning: Minimizes BCE for encoded real pairs labeled one
            and generated pairs labeled zero in the BiGAN discriminator game.
        """
        with torch.no_grad():
            real_latents = self.encoder(observations)
            latent_samples = self._sample_latents(observations.shape[0])
            fake_observations = self.generator(latent_samples)
        real_logits = self.discriminator(observations, real_latents)
        fake_logits = self.discriminator(fake_observations, latent_samples)
        loss = discriminator_loss(real_logits, fake_logits)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.detach(), real_logits.detach().mean(), fake_logits.detach().mean()

    def _encoder_generator_step(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Perform one joint encoder/generator adversarial update.

        Input: A floating-point replay batch of real observations.
        Output: ``(adversarial_loss, reconstruction_loss)`` detached scalars.
        Mathematical meaning: Optimizes only the non-saturating BiGAN
            adversarial objective from Eq. (3). Reconstruction error is
            computed afterward as a detached diagnostic/scoring statistic and
            does not contribute gradients.
        """
        self._set_requires_grad(self.discriminator, False)
        try:
            real_latents = self.encoder(observations)
            latent_samples = self._sample_latents(observations.shape[0])
            fake_observations = self.generator(latent_samples)
            generated_latents = latent_samples
            real_logits = self.discriminator(observations, real_latents)
            fake_logits = self.discriminator(fake_observations, generated_latents)
            adversarial = encoder_generator_loss(real_logits, fake_logits)
            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.generator_optimizer.zero_grad(set_to_none=True)
            # Eq. (3): only the BiGAN adversarial objective is optimized.
            adversarial.backward()
            if not self.encoder_frozen:
                self.encoder_optimizer.step()
            self.generator_optimizer.step()
            with torch.no_grad():
                diagnostic_latents = self.encoder(observations)
                diagnostic_reconstructions = self.generator(diagnostic_latents)
                reconstruction = pixel_reconstruction_loss(
                    observations,
                    diagnostic_reconstructions,
                )
        finally:
            self._set_requires_grad(self.discriminator, True)
        return adversarial.detach(), reconstruction.detach()

    def update(self, replay_buffer: ReplayBuffer) -> BiGANUpdateMetrics:
        """Run the configured BiGAN schedule if replay is ready.

        Input: Replay buffer containing at least ``config.batch_size`` valid
            observations.
        Output: ``BiGANUpdateMetrics``. ``updated=False`` indicates warmup or
            interval gating and no optimizer step was performed.
        Mathematical meaning: Executes alternating optimization of the
            discriminator and encoder/generator objectives for the current
            empirical observation distribution.
        """
        if self.update_count < self.config.warmup_updates:
            self.update_count += 1
            return BiGANUpdateMetrics(False, 0.0, 0.0, 0.0, 0.0, 0.0, self.update_count)
        if (self.update_count - self.config.warmup_updates) % self.config.update_interval != 0:
            self.update_count += 1
            return BiGANUpdateMetrics(False, 0.0, 0.0, 0.0, 0.0, 0.0, self.update_count)
        if len(replay_buffer) < self.config.batch_size:
            return BiGANUpdateMetrics(False, 0.0, 0.0, 0.0, 0.0, 0.0, self.update_count)

        discriminator_values: list[Tensor] = []
        real_logits: list[Tensor] = []
        fake_logits: list[Tensor] = []
        encoder_generator_values: list[Tensor] = []
        reconstruction_values: list[Tensor] = []
        for _ in range(self.config.discriminator_steps):
            observations = self._model_observations(
                replay_buffer.sample(self.config.batch_size, device=self.device).observations
            )
            loss, real_logit, fake_logit = self._discriminator_step(observations)
            discriminator_values.append(loss)
            real_logits.append(real_logit)
            fake_logits.append(fake_logit)
        for _ in range(self.config.generator_encoder_steps):
            observations = self._model_observations(
                replay_buffer.sample(self.config.batch_size, device=self.device).observations
            )
            adversarial, reconstruction = self._encoder_generator_step(observations)
            encoder_generator_values.append(adversarial)
            reconstruction_values.append(reconstruction)

        self.update_count += 1
        return BiGANUpdateMetrics(
            updated=True,
            discriminator_loss=float(torch.stack(discriminator_values).mean().cpu()),
            encoder_generator_loss=float(torch.stack(encoder_generator_values).mean().cpu()),
            reconstruction_loss=float(torch.stack(reconstruction_values).mean().cpu()),
            discriminator_real_logit=float(torch.stack(real_logits).mean().cpu()),
            discriminator_fake_logit=float(torch.stack(fake_logits).mean().cpu()),
            update_count=self.update_count,
        )
