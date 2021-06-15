from _future_ import print_function
from _future_ import division
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss
from torch import nn
from torch.utils.data import DataLoader
import wandb
from Classes import EmotionData, EmotionDataTest
from torch.utils.data.dataloader import default_collate
from ignite.handlers import ModelCheckpoint
from ignite.handlers import EarlyStopping



wandb.login(key="a8895ab6bdbe3827b2a137c581e59e9440154140") # this is my api, subscribe on wandb and paste your API here to monitor your training


# Load the pretrained model from pytorch
model = models.vgg16(pretrained=True)
print(model.classifier[6].out_features) # 1000
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 7)]) # Add our layer with 7 outputs
model.classifier = nn.Sequential(*features)

for param in model.features.parameters():
    param.require_grad = True

print(model)



lr = 1e-4

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
TrainLoader = DataLoader(TrainData, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)  # create train data loaders

TestData = EmotionDataTest('test.csv', './')
TestLoader = DataLoader(TestData, batch_size=64, pin_memory=True,num_workers=4)  # create validation data loaders

#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)


criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion,
                                    device='cuda:0' if torch.cuda.is_available() else 'cpu')

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
#evaluator = create_supervised_evaluator(model, metrics=val_metrics)
train_evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                    device='cuda:0' if torch.cuda.is_available() else 'cpu')
validation_evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                    device='cuda:0' if torch.cuda.is_available() else 'cpu')


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

# Add checkpoint
checkpointer = ModelCheckpoint('saved_models', 'VGG16e-4', n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'VGG16e-4': model})






def score_function(trainer):
    val_loss = trainer.state.metrics['loss']
    return -val_loss

handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
# Note: the handler is attached to an Evaluator (runs one epoch on validation dataset).
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

trainer.run(TrainLoader, max_epochs=30)