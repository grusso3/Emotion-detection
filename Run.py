import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch import nn
import wandb
from Classes import Network
from Classes import EmotionData
lr = 0.001
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

model = Network()

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)


@trainer.on(Events.ITERATION_COMPLETED(every=10))
def log_training_loss(trainer):

    wandb.log({"loss": trainer.state.output})

    print(f"Epoch[{trainer.state.epoch}] | Iter[{trainer.state.iteration}] | Loss: {trainer.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(TrainLoader)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(TestLoader)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


trainer.run(TrainLoader, max_epochs=10)
