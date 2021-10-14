import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from config import (
    batch_size,
    latent_dim,
    n_samples,
    n_steps,
    sample_interval
)

from utils.helpers import (
    generate_multitrack,
    generate_multitrack_conditional,
    save_models,
    save_midi_sample
)

class ConditionalModelTrainer:
    def __init__(
        self,
        data_loader: DataLoader,
        discriminator,
        generator
    ):
        # Data loader
        self.data_loader = data_loader

        # creating models
        self.discriminator = discriminator
        self.generator = generator

        self.sample_latent = torch.randn(n_samples, latent_dim)

        if torch.cuda.is_available():
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()
            self.sample_latent = self.sample_latent.cuda()

        # creating optimizers
        self.d_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
        self.g_optimizer = torch.optim.Adam(
            generator.parameters(), lr=0.001, betas=(0.5, 0.9))

        self.step = 0

    
    def _compute_gradient_penalty(self, real_samples, fake_samples, labels):
        """Compute the gradient penalty for regularization. Intuitively, the
        gradient penalty help stablize the magnitude of the gradients that the
        discriminator provides to the generator, and thus help stablize the training
        of the generator."""
        # Get random interpolations between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1)
        if torch.cuda.is_available():
            alpha = alpha.cuda()
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
        interpolates = interpolates.requires_grad_(True)
        # Get the discriminator output for the interpolations
        d_interpolates = self.discriminator(interpolates, labels)
        # Get gradients w.r.t. the interpolations
        fake = torch.ones(real_samples.size(0), 1)
        if torch.cuda.is_available():
            fake = fake.cuda()
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    
    def _train_one_step(self, real_samples, real_label):
        """Train the networks for one step."""
        # Sample from the lantent distribution
        latent = torch.randn(batch_size, latent_dim)

        # Transfer data to GPU
        if torch.cuda.is_available():
            real_samples = real_samples.cuda()
            latent = latent.cuda()
        
        # === Train the discriminator ===
        # Reset cached gradients to zero
        self.d_optimizer.zero_grad()
        # Get discriminator outputs for the real samples
        prediction_real = self.discriminator(real_samples, real_label)
        # Compute the loss function
        # d_loss_real = torch.mean(torch.nn.functional.relu(1. - prediction_real))
        d_loss_real = -torch.mean(prediction_real)
        # Backpropagate the gradients
        d_loss_real.backward()
        
        # Generate fake samples with the generator
        fake_samples = self.generator(latent, real_label)
        # Get discriminator outputs for the fake samples
        prediction_fake_d = self.discriminator(fake_samples.detach(), real_label)
        # Compute the loss function
        # d_loss_fake = torch.mean(torch.nn.functional.relu(1. + prediction_fake_d))
        d_loss_fake = torch.mean(prediction_fake_d)
        # Backpropagate the gradients
        d_loss_fake.backward()

        # Compute gradient penalty
        gradient_penalty = 10.0 * self._compute_gradient_penalty(real_samples.data, fake_samples.data, real_label.data)
        # Backpropagate the gradients
        gradient_penalty.backward()

        # Update the weights
        self.d_optimizer.step()
        
        # === Train the generator ===
        # Reset cached gradients to zero
        self.g_optimizer.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples, real_label)
        # Compute the loss function
        g_loss = -torch.mean(prediction_fake_g)
        # Backpropagate the gradients
        g_loss.backward()
        # Update the weights
        self.g_optimizer.step()

        return d_loss_real + d_loss_fake, g_loss


    def train(self):
        # Create a progress bar instance for monitoring
        progress_bar = tqdm(total=n_steps, initial=self.step, ncols=80, mininterval=1)

        # Start iterations
        while self.step < n_steps + 1:
            # Iterate over the dataset
            for real_samples, real_label in self.data_loader:
                # Train the neural networks
                self.generator.train()
                d_loss, g_loss = self._train_one_step(real_samples, real_label)

                # Record smoothened loss values to LiveLoss logger
                if self.step > 0:
                    running_d_loss = 0.05 * d_loss + 0.95 * running_d_loss
                    running_g_loss = 0.05 * g_loss + 0.95 * running_g_loss
                else:
                    running_d_loss, running_g_loss = 0.0, 0.0

                # Update losses to progress bar
                progress_bar.set_description_str(
                    "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))
                
                if self.step % sample_interval == 0:
                    save_models(self.generator, self.discriminator, self.step, "sagan")

                    self.generator.eval()
                    latent_sample = torch.randn(1, latent_dim)
                    random_number = random.randint(0, 12)
                    label = torch.zeros(1, 13)
                    label[0][random_number] = 1.0
                    if torch.cuda.is_available():
                        latent_sample = latent_sample.cuda()
                    multitrack = generate_multitrack_conditional(self.generator, latent_sample, label)
                    save_midi_sample(multitrack, "sagan", self.step)
                    
                self.step += 1
                progress_bar.update(1)
                if self.step >= n_steps:
                    break
        save_models(self.generator, self.discriminator, self.step, "sagan")

        self.generator.eval()
        latent_sample = torch.randn(1, latent_dim)
        if torch.cuda.is_available():
            latent_sample = latent_sample.cuda()
        multitrack = generate_multitrack(self.generator, latent_sample)
        save_midi_sample(self.generator, "attention", self.step)

# packages for main
from models.sagan.discriminator import Discriminator
from models.sagan.generator import Generator
from utils.conditional_data_loader import CreateDataLoader


if __name__ == "__main__":
    # building the models
    generator = Generator()
    discriminator = Discriminator()

    # getting the data loader
    data_loader = CreateDataLoader().load_data_loader()

    #building the trainer
    trainer = ConditionalModelTrainer(data_loader, discriminator, generator)

    #training the model
    trainer.train()
