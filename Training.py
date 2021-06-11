import torch
from torch import optim, nn
from Classes import *
import torch.nn.functional as F
from torch.utils.data import DataLoader




# Define function that returns correct number of predicted labels and accuracy

def get_correct_number(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum().item()

# get accuracy
def accuracy(pred, labels):
    classes = torch.argmax(pred, dim=1)
    return torch.mean((classes == labels).float())





network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# with torch.no_grad():
#     train_preds = get_pred(network, Train_Loader)


TrainData = EmotionData('train.csv', './')
TrainLoader = DataLoader(TrainData, batch_size=32, shuffle=True, pin_memory=True)  # create train data loaders

TestData = EmotionData('train.csv', './')
TestLoader = DataLoader(TestData, batch_size=64, pin_memory=True)  # create validation data loaders


for epoch in range(10):
    total_loss = 0
    total_correct = 0
    total_accuracy = 0
    total_t = 0
    train_epoch_acc = 0
    for batch in TrainLoader:
        images, labels = batch
        pred = network(images)

        loss = criterion(pred, labels)  # calculate the loss
        #loss = criterion(pred, torch.max(labels)  # calculate the loss
        optimizer.zero_grad()  # Before calculating the gradient we need to zero out currently existing gradients
        loss.backward()  # calculate Gradients
        optimizer.step()  # Update weights

        total_loss += loss.item()
        total_correct += get_correct_number(pred, labels)
        total_accuracy += accuracy(pred, labels)

    print("epoch", epoch, " | ", "total_correct:", total_correct, " | ", "loss", total_loss, " | ", "total accuracy", total_accuracy / len(TrainLoader),"let's get a 6 ",train_epoch_acc)
# ------------------------------------



total_accuracy / len(TrainLoader)  # 25 % accuracy

# Get the gradient value of each layer
network.conv1.weight.max().item()
network.conv2.weight.max().item()
network.fc1.weight.max().item()
network.fc2.weight.max().item()

# We basically are underfitting our model and thus, we need to change the architecture of our network as well as tuning the hyperparameters




