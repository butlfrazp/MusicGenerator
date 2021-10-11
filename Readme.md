# Music Generator

## Goal

The goal of the project is to test different implementations of generating music using GANs (generative adversarial networks).
There is a baseline model that is provided utilizing convolutional neural networks.
This model is provided by [Hao-Wen Dong et. al](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf) as an extension of their MuseGAN work.
We are proposing exploring techniques utilizing transformers and attentive models to increase the quality of the generated models.
We are also proposing automating our training to be run on AML (honestly, I don't really care about this, but if people want to then dope).

## Running the project

### Creating the dataset

To get the dataset run `./scripts/create_dataset.sh`.

### Training the model

To train the model run `python -m main`

## Helpful links

- [MuseGAN](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf)
- [Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/dataset)
