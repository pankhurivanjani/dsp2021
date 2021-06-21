import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import pdb

# 2.1  
wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names) 
wine_df = pd.DataFrame(wine_data)
#[178 rows x 14 columns]
#wine_df['Class'].nunique() - 3
features_df = wine_df.drop(['Class'], axis=1)
features_df = (features_df - features_df.mean()) / features_df.std()

# 2.2
def get_n_pca(min_cum_variances, data_frame):
    data = data_frame.values
    cov = np.cov(data.T)
    ev, _ = np.linalg.eig(cov)
    for n_pca in range(1, data.shape[1]):
        cum_variances = np.sort(ev)[-n_pca:].sum() / ev.sum()
        if cum_variances >= min_cum_variances:
            return n_pca

def get_cum_variances(n_pca, data_frame):
    data = data_frame.values
    cov = np.cov(data.T)
    ev, _ = np.linalg.eig(cov)
    cum_variances = np.sort(ev)[-n_pca:].sum() / ev.sum()
    return cum_variances

# 2.3 
n_pca = 3
cum_variances = get_cum_variances(n_pca, features_df)

cum_variances = 0.5
n_pca = get_n_pca(cum_variances, features_df)