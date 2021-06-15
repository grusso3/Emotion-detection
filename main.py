import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss
from torch import nn
from torch.utils.data import DataLoader
import wandb
from Classes import Network, Network2
from Classes import EmotionData,EmotionDataTest
from ignite.handlers import ModelCheckpoint, EarlyStopping
import hiddenlayer as hl

model = Network()

hl.build_graph(model, torch.zeros([32, 3, 48, 48]))

lr = 1e-6
wandb.login(key="a8895ab6bdbe3827b2a137c581e59e9440154140") # this is my api, subscribe on wandb and paste your API here to monitor your training

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

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")


model = model.to(device)

TrainData = EmotionData('train.csv', './')
TrainLoader = DataLoader(TrainData, batch_size=32, shuffle=True, pin_memory=True,num_workers=4)  # create train data loaders

TestData = EmotionDataTest('test.csv', './')
TestLoader = DataLoader(TestData, batch_size=64, pin_memory=True,num_workers=4)  # create validation data loaders

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion,
                                    device= "cuda:0" if torch.cuda.is_available() else "cpu")

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
#evaluator = create_supervised_evaluator(model, metrics=val_metrics)
train_evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                    device= "cuda:0" if torch.cuda.is_available() else "cpu")
validation_evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                    device= "cuda:0" if torch.cuda.is_available() else "cpu")


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(trainer):

    #wandb.log({"loss": trainer.state.output})
    print(f"Epoch[{trainer.state.epoch}] | Iter[{trainer.state.iteration}] | Loss: {trainer.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(TrainLoader)
    metrics = train_evaluator.state.metrics
    wandb.log({"Train loss":   metrics['loss'],
               "Train Accuracy":   metrics['accuracy']})

    print(
        f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:} Avg loss: {metrics['loss']:}| Train Loss: {trainer.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):

    validation_evaluator.run(TestLoader)
    metrics = validation_evaluator.state.metrics
    wandb.log({"Val loss":  metrics['loss'],
               "Val Accuracy":   metrics['accuracy']})

    print(
        f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:} Avg loss: {metrics['loss']:} | Val Loss: {trainer.state.output:.2f}")


checkpointer = ModelCheckpoint('saved_models', 'EmotionData', n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'EmotionData': model})



trainer.run(TrainLoader, max_epochs=30)