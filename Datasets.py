import os
import matplotlib.pyplot as plt
import pandas as pd

import sys
base_path = './Data/archive'
# create csv file from the data
for kind in ['train', 'test']:
    df = dict(image=[], label=[])
    sub_path = os.path.join(base_path, kind)
    classes = os.listdir(sub_path)

    for class_ in classes:
        if '.' in class_:
            continue
        images = os.path.join(sub_path, class_)
        for image in os.listdir(images):
            if image.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                continue
            df['image'].append(os.path.join(images, image))
            df['label'].append(class_)

#
    x = pd.DataFrame(df)
    print(x.groupby("label").count())
    x.label = pd.Categorical(pd.factorize(x.label)[0])
    x.to_csv(f'{kind}.csv', index=False)


# visualizing data labels
plt.rcParams['figure.figsize'] = (15,5)
x['label'].value_counts().plot(kind = 'bar')
plt.title("Emotions detected", fontsize = 20 )
plt.show()