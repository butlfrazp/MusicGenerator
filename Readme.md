# Music Generator

## Goal

The goal of the project is to test different implementations of generating music using GANs (generative adversarial networks).
There is a baseline model that is provided utilizing convolutional neural networks.
This model is provided by [Hao-Wen Dong et. al](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf) as an extension of their MuseGAN work.
We are proposing exploring techniques utilizing transformers and attentive models to increase the quality of the generated models.
We are also proposing automating our training to be run on AML (honestly, I don't really care about this, but if people want to then dope).

## Running the project

### Install Requirements

Activate your python virtual environment and then install the requirements
by running `pip install -r requirements.txt`

### Creating the dataset

To get the dataset run `./scripts/create_dataset.sh`.

### Training the model

To train the model run `python -m main`.

If you do this on your local machine it will take ~2.5 hours.

### Generating Music

Once you have a trained model to use run `python -m sample_generator.base_model_sample_generator` to generate music.

### Running the sample

To run the sample run `python -m sample_generator.base_model_sample_generator`.
The output will show the name of the file that the program generated.

## Helpful links

- [MuseGAN](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf)
- [Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/dataset)
