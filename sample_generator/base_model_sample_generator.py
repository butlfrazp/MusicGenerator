import torch

from config import (
    latent_dim,
    n_pitches,
    n_tracks
)
from models.base.discriminator import Discriminator
from models.base.generator import Generator
from utils.helpers import generate_multitrack, load_models, save_midi_sample

class BaseModelSampleGenerator:
    def __init__(self, training_iter: int):
        _, self.generator = load_models(training_iter)

    def generate_sample(self):
        sample_latent = torch.randn(1, latent_dim)
        
        multitrack = generate_multitrack(self.generator, sample_latent)

        file_name = save_midi_sample(multitrack)
        print(f"Create Sample: {file_name};")
    
if __name__ == "__main__":
    training_iter = 20000
    sampler = BaseModelSampleGenerator(training_iter)
    sampler.generate_sample()

