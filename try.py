#%%
import torch

#%%
from torch import optim
from main import *
from torch.utils.data import TensorDataset, DataLoader
from Classes import Network
import torch.nn.functional as F
from Run import *

#%%
# let's try our dataset through the Network
batch = next(iter(Train_Loader))  # Get Batch
images, label = batch

# Define network
network = Network()
pred = network(images)  # Pass Batch
print(pred)

# Play a bit with the predictions

pred.argmax(dim=1)  # see the label that has more probability
pred.argmax(dim=1).eq(label)  # this gives us a 1 if the predicted category matches the label and 0 otherwise
pred.argmax(dim=1).eq(label).sum()  # correct number of predictions


# Define function that returns correct number of predicted labels

def get_correct_number(pred, label):
    return pred.argmax(dim=1).eq(label).sum().item()


# Compute Loss function
loss = F.cross_entropy(pred, torch.max(label, 1)[1])
loss.item()
print(network.conv1.weight.grad)  # returns none

# Compute the gradient; per each parameter in the weight tensor there's a corresponding gradient
loss.backward()
network.conv1.weight.grad.shape

# what happened here? When we pass our images through our network, these flow through the function defined in our
# forward. meanwhile pytorch is keeping track of these calculations. When we call the prediction tensor, because the
# latter come from our netwrok, the prediction tensor has all of the previuous calculation. afterwards we use the
# pred tensor to compute the loss tensor. When we call backward on the loss tensor all of the gradients per each
# tensor can be calculated.

# Now using these gradients we're going to update the weights; to do that we're using adam Optimizer

optimizer = optim.Adam(network.parameters(), lr=0.01)
loss.item()  # check the loss
optimizer.step()  # update the weights towards the loss function's minimum
pred = network(images)
loss = F.cross_entropy(pred, torch.max(label, 1)[1])  # calculate the loss
loss.item()  # now we can see ho much our loss decreased. As from 2.06 to 1.3, that's huge!

# =================================
# Summarizing what we've done so far
# =================================
network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)

batch = next(iter(Train_Loader))  # Get Batch
images, label = batch

pred = network(images)
loss = F.cross_entropy(pred, torch.max(label, 1)[1])  # calculate the loss

loss.backward()  # calculate Gradients
optimizer.step()  # Update weights

# ------------------------------------
print("loss1:", loss.item())
pred = network(images)
loss = F.cross_entropy(pred, torch.max(label, 1)[1])
print("loss2:", loss.item())
