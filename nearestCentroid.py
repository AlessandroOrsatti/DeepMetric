import pandas as pd
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from tqdm import tqdm
import random
import utils
import matplotlib.pyplot as plt
import os
import parameters as param


df = pd.read_csv(os.path.join(param.results_dir, "results_modified_dataset2.csv"))
df.sort_values('ID')

original_list = df['original'].values.tolist()
original_set = set(original_list)
original_list = list(original_set)
original_list.remove('-')
random.shuffle(original_list)
del original_set


y_train = []
y_test = []

x_train_c = []
x_test_c = []
x_train_e = []
x_test_e = []
x_train_m = []
x_test_m = []

original_train = original_list[0:int(len(original_list)/2)]
original_test = original_list[int(len(original_list)/2):]

for i in tqdm(original_train, total=len(original_train)):

    original = df[df['ID']==i]
    temp = df[df['original']==i]
    temp = temp.sort_values('ID')
    temp_c = temp['cosine distance'].to_numpy()
    temp_e = temp['euclidean distance'].to_numpy()
    temp_m = temp['manhattan distance'].to_numpy()
    if len(temp) == 15:
        y_train.append(original['label'])
        x_train_c.append(temp_c)
        x_train_e.append(temp_e)
        x_train_m.append(temp_m)

for i in tqdm(original_test, total=len(original_test)):

    original = df[df['ID'] == i]
    temp = df[df['original'] == i]
    temp = temp.sort_values('ID')
    temp_c = temp['cosine distance'].to_numpy()
    temp_e = temp['euclidean distance'].to_numpy()
    temp_m = temp['manhattan distance'].to_numpy()
    if len(temp) == 15:
        y_test.append(original['label'])
        x_test_c.append(temp_c)
        x_test_e.append(temp_e)
        x_test_m.append(temp_m)

y_train = np.array(y_train)
y_train = y_train.flatten()
y_test = np.array(y_test)
y_test = y_test.flatten()

x_train_c = np.vstack(x_train_c)
x_test_c = np.vstack(x_test_c)
x_train_e = np.vstack(x_train_e)
x_test_e = np.vstack(x_test_e)
x_train_m = np.vstack(x_train_m)
x_test_m =  np.vstack(x_test_m)

clf_c = NearestCentroid()
clf_e = NearestCentroid()
clf_m = NearestCentroid()
clf_c.fit(x_train_c, y_train)
clf_e.fit(x_train_e, y_train)
clf_m.fit(x_train_m, y_train)

results_c = clf_c.predict(x_test_c)
results_e = clf_e.predict(x_test_e)
results_m = clf_m.predict(x_test_m)

df_filtered = df[df['original'] == '-']
plt.figure(figsize=(8, 8))
utils.plot_roc_curve(df_filtered['label'], df_filtered['prediction'], legend='Original\ score')
utils.plot_roc_curve(y_test, results_c, legend='Cosine\ distance')
utils.plot_roc_curve(y_test, results_e, legend='Euclidean\ distance')
utils.plot_roc_curve(y_test, results_m, legend='Manhattan\ distance')
plt.savefig('rocCentroid.pdf')
plt.show()
