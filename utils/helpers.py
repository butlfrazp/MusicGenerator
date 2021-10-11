import numpy as np
import os
import pypianoroll
from pypianoroll import (
    Multitrack,
    StandardTrack
)
import torch
from uuid import uuid4

from config import (
    beat_resolution,
    is_drums,
    lowest_pitch,
    midi_output_dir,
    model_output_dir,
    n_pitches,
    n_tracks,
    programs,
    tempo_array,
    track_names
)
from models.base.generator import Generator
from models.base.discriminator import Discriminator


def generate_multitrack(generator, sample_latent) -> Multitrack:
    # Get generated samples
    generator.eval()
    samples = generator(sample_latent).cpu().detach().numpy()

    samples = samples.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(programs, is_drums, track_names)
    ):
        pianoroll = np.pad(
            samples[idx] > 0.5,
            ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
        )
        tracks.append(
            StandardTrack(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll
            )
        )
    m = Multitrack(
        tracks=tracks,
        tempo=tempo_array,
        resolution=beat_resolution
    )

    return m


def _save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def load_models(training_iter, model_type="base") -> (Discriminator, Generator):
    model_type_path = os.path.join(model_output_dir, model_type)
    generator_path = os.path.join(model_type_path, "generator", f"generator_{training_iter}.pth")
    discriminator_path = os.path.join(model_type_path, "discriminator", f"discriminator_{training_iter}.pth")

    if not (os.path.exists(generator_path) and os.path.exists(discriminator_path)):
        raise Exception(f"""
        Path does not exist.
        Generator Path: {generator_path};
        Discriminator Path: {discriminator_path};
        """)

    generator = Generator()
    generator.load_state_dict(torch.load(generator_path))

    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(discriminator_path))

    return discriminator, generator


def save_models(generator, discriminator, training_iter, model_type="base"):
    model_type_path = os.path.join(model_output_dir, model_type)
    generator_path = os.path.join(model_type_path, "generator")
    discriminator_path = os.path.join(model_type_path, "discriminator")

    if not os.path.exists(model_type_path):
        os.mkdir(model_type_path)
        os.mkdir(generator_path)
        os.mkdir(discriminator_path)
    
    generator_filename = os.path.join(generator_path, f"generator_{training_iter}.pth")
    discriminator_filename = os.path.join(discriminator_path, f"discriminator_{training_iter}.pth")

    _save_model(generator, generator_filename)
    _save_model(discriminator, discriminator_filename)


def save_midi_sample(multitrack, model_type="base") -> str:
    midi_type_path = os.path.join(midi_output_dir, model_type)

    if not os.path.exists(midi_type_path):
        os.mkdir(midi_type_path)

    sample_id = uuid4()
    sample_id = str(sample_id)[:8]

    file_name = f"sample_{sample_id}.mid"

    out_file = os.path.join(midi_type_path, file_name)

    pypianoroll.write(out_file, multitrack=multitrack)

    return file_name