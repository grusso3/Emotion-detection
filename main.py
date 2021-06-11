import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss
from torch import nn
from torch.utils.data import DataLoader
import wandb
from Classes import Network, Network2
from Classes import EmotionData


lr = 0.01

wandb.init(
      # Set entity to specify your username or team name
      # ex: entity="carey",
      # Set the project where this run will be logged
      project="EmotionDetection",
      # Track hyperparameters and run metadata
      config={
      "learning_rate": lr,
      "architecture": "CNN",
      "dataset": "EmotionDataset"})

model = Network2()

TrainData = EmotionData('train.csv', './')
TrainLoader = DataLoader(TrainData, batch_size=32, shuffle=True, pin_memory=True)  # create train data loaders

TestData = EmotionData('test.csv', './')
TestLoader = DataLoader(TestData, batch_size=64, pin_memory=True)  # create validation data loaders

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
#evaluator = create_supervised_evaluator(model, metrics=val_metrics)
train_evaluator = create_supervised_evaluator(model, metrics=val_metrics)
validation_evaluator = create_supervised_evaluator(model, metrics=val_metrics)


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(trainer):

    #wandb.log({"loss": trainer.state.output})
    print(f"Epoch[{trainer.state.epoch}] | Iter[{trainer.state.iteration}] | Loss: {trainer.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(TrainLoader)
    metrics = train_evaluator.state.metrics
    wandb.log({"Train loss":   metrics['loss']})

    print(
        f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:} Avg loss: {metrics['loss']:}| Train Loss: {trainer.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):

    validation_evaluator.run(TestLoader)
    metrics = validation_evaluator.state.metrics
    wandb.log({"Val loss":  metrics['loss']})

    print(
        f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:} Avg loss: {metrics['loss']:} | Val Loss: {trainer.state.output:.2f}")


trainer.run(TrainLoader, max_epochs=25)

from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

# show_batch(Train_Loader)
