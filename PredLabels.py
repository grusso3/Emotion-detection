from main import *
import pathlib
import numpy as np
import pandas as pd
from main import TestLoader



model_path = next(pathlib.Path('saved_models').rglob('*'))
model_path

model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)

predictions = []
labels = []

# change model mode to 'evaluation'
# disable dropout and use learned batch norm statistics
model.eval()

with torch.no_grad():
    for batch in TestLoader:
        x, label = batch
#         logits = model(title)
        logits = model(x)

        y_pred = torch.max(logits, dim=1)[1]
        # move from GPU to CPU and convert to numpy array
        y_pred_numpy = y_pred.cpu().numpy()

        predictions = np.concatenate([predictions, y_pred_numpy])

classes = ['angry', 'disgust', 'fear',"happy","neutral","sad","surprise"]

predictions_str = [classes[int(p)] for p in predictions]
# test.csv index in a contiguous integers from 0 to len(test_set)
# to this should work fine
submission = pd.DataFrame({'id': list(range(len(predictions_str))), 'label': predictions_str})

submission.head()