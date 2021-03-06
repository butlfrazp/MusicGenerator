from models.base.discriminator import Discriminator
from models.base.generator import Generator
from trainers.base_model_trainer import BaseModelTrainer
from utils.data_loader import CreateDataLoader

if __name__ == "__main__":
    # building the models
    generator = Generator()
    discriminator = Discriminator()

    # getting the data loader
    data_loader = CreateDataLoader().load_data_loader()

    #building the trainer
    trainer = BaseModelTrainer(data_loader, discriminator, generator)

    #training the model
    trainer.train()
