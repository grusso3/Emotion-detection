from main import *
from Classes import *
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# loading the saved model
def fetch_last_checkpoint_model_filename(model_save_path):
    import os
    checkpoint_files = os.listdir(model_save_path)
    checkpoint_files = [f for f in checkpoint_files if '.pt' in f]
    checkpoint_iter = [
        int(x.split('_')[2].split('.')[0])
        for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()
    return os.path.join(model_save_path, checkpoint_files[last_idx])


model.load_state_dict(torch.load(fetch_last_checkpoint_model_filename('saved_models')))
print("Model Loaded")



# classes of fashion mnist dataset
classes = ['angry', 'disgust', 'fear',"happy","neutral","sad","surprise"]
# creating iterator for iterating the dataset
dataiter = iter(TestLoader)
images, labels = dataiter.next()
images_arr = []
labels_arr = []
pred_arr = []
# moving model to cpu for inference
model.to("cpu")
# iterating on the dataset to predict the output
for i in range(0,7):
    images_arr.append(images[i].unsqueeze(0))
    labels_arr.append(labels[i].item())
    ps = torch.exp(model(images_arr[i]))
    ps = ps.data.numpy().squeeze()
    pred_arr.append(np.argmax(ps))
# plotting the results
fig = plt.figure(figsize=(25,4))
for i in range(7):
    ax = fig.add_subplot(2, 7, i+1, xticks=[], yticks=[])
    ax.imshow(images_arr[i].resize_(1, 48, 48).numpy().squeeze())
    ax.set_title("{} ({})".format(classes[pred_arr[i]], classes[labels_arr[i]]),
                 color=("green" if pred_arr[i]==labels_arr[i] else "red"))
plt.show()


#