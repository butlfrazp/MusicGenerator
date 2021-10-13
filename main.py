from models.base.discriminator import Discriminator
from models.base.generator import Generator
from trainers.model_trainer import ModelTrainer
from utils.data_loader import CreateDataLoader

if __name__ == "__main__":
    # building the models
    generator = Generator()
    discriminator = Discriminator()

    # getting the data loader
    print("Loading data")
    data_loader = CreateDataLoader().load_data_loader()

    #building the trainer
    trainer = ModelTrainer(data_loader, discriminator, generator)

    #training the model
    print("Training model")
    trainer.train()
