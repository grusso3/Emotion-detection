import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torch
from sklearn.metrics import confusion_matrix
from torchvision import models
from Classes import EmotionDataTest
from torch.utils.data import DataLoader


TestData = EmotionDataTest('test.csv', './')
TestLoader = DataLoader(TestData, batch_size=64, pin_memory=True,num_workers=4)  # create validation data loaders

########### BRUTE FORCE #########
model16 = models.vgg16(pretrained=True)
# Newly created modules have require_grad=True by default
num_features = model16.classifier[6].in_features
features = list(model16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 7)]) # Add our layer with 4 outputs
model16.classifier = nn.Sequential(*features)
model16.load_state_dict(torch.load("saved_models/VGG16e-4_VGG16e-4_8082.pt", map_location=torch.device('cpu')),strict= False)

model16.eval()


prediction = []
label = []

correct, all = 0,0
for images,labels in TestLoader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        img = images[i].view(1,3,48,48)
        with torch.no_grad():
            logps = model16(img)
        ps = torch.exp(logps)
        prob = list(ps.cpu()[0])
        pred_label = prob.index(max(prob))
        true_label = labels.cpu()[i]
        if (true_label == pred_label):
            correct += 1
        all += 1
        prediction.append(pred_label)
        label.append(true_label)
print("Number of images", all)
print("Accuracy",correct/all)

cm = confusion_matrix(label,prediction)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['angry', 'disgust', 'fear',"happy","neutral","sad","surprise"]); ax.yaxis.set_ticklabels(['angry', 'disgust', 'fear',"happy","neutral","sad","surprise"]);
plt.show()
df = pd.DataFrame({"Actual" : label,"Predicted":prediction})
df


#################### VGG 11
model11 = models.vgg11(pretrained=True)
# Newly created modules have require_grad=True by default
num_features = model11.classifier[6].in_features
features = list(model11.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 7)]) # Add our layer with 4 outputs
model11.classifier = nn.Sequential(*features)
model11.load_state_dict(torch.load("saved_models/VGG16e-4_VGG16e-4_8082.pt"),strict= False)

model16.eval()


prediction = []
label = []

correct, all = 0,0
for images,labels in TestLoader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        img = images[i].view(1,3,48,48)
        with torch.no_grad():
            logps = model11(img)
        ps = torch.exp(logps)
        prob = list(ps.cpu()[0])
        pred_label = prob.index(max(prob))
        true_label = labels.cpu()[i]
        if (true_label == pred_label):
            correct += 1
        all += 1
        prediction.append(pred_label)
        label.append(true_label)
print("Number of images", all)
print("Accuracy",correct/all)

cm11 = confusion_matrix(label,prediction)

cm11 = confusion_matrix(label,prediction)
ax= plt.subplot()
sns.heatmap(cm11, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['angry', 'disgust', 'fear',"happy","neutral","sad","surprise"]); ax.yaxis.set_ticklabels(['angry', 'disgust', 'fear',"happy","neutral","sad","surprise"]);
plt.show()

df11 = pd.DataFrame({"Actual" : label,"Predicted":prediction})
df11

