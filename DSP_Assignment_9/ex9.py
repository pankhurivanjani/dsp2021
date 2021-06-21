import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy import linalg as LA

# 2.1  
wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names) 
wine_df = pd.DataFrame(wine_data)
#[178 rows x 14 columns]
#wine_df['Class'].nunique() - 3
features_df = wine_df.drop(['Class'], axis=1)

# 2.2
def PCA(orig_data):
    #orig_data = data_frame.values # pandas to numpy array
    pdb.set_trace()
    cov = np.cov(orig_data, rowvar=False)
    #pdb.set_trace()
    #from matplotlib.mlab import PCA
    #results = PCA(data)
    eig_vals, eig_vectors = LA.eigh(cov)#np.linalg.eig(cov)
    idx = np.argsort(eig_vals)[::-1] # sort eigenvalues in decreasing order
    eig_vals = eig_vals[idx] # re-order eigenvalues
    eig_vectors = eig_vectors[:, idx] # re-order eigenvectors
    #recovered_data = np.dot(eig_vectors.T, orig_data.T).T
    recovered_data = np.dot(orig_data, eig_vectors)
    return recovered_data, eig_vals, eig_vectors

'''
pca = decomposition.PCA(n_components=13)
features_df_std_auto = StandardScaler().fit_transform(features_df)
features_df_pca_auto = pca.fit_transform(features_df_std_auto)
'''
pdb.set_trace()
features_df_std_manual = (features_df - features_df.mean()) / features_df.std()
features_df_pca_manual, _, _ = PCA(features_df_std_manual.values)
#print(np.linalg.norm(features_df_pca_auto - features_df_pca_manual))

def get_n_pca(min_cum_variances, data_frame): #TODO verify with scipy # TODO PCA as a separate function
    orig_data = data_frame.values # pandas to numpy array
    cov = np.cov(orig_data.T)
    pdb.set_trace()
    #from matplotlib.mlab import PCA
    #results = PCA(data)
    ev, eig = np.linalg.eig(cov)
    idx = np.argsort(ev)[::-1] # sort eigenvalues in decreasing order
    eig = eig[:, idx] # re-order eigenvectors
    ev = ev[idx] # re-order eigenvalues
    recovered_data = np.dot(eig.T, orig_data.T).T
    for n_pca in range(1, orig_data.shape[1]):
        cum_variances = np.sort(ev)[-n_pca:].sum() / ev.sum()
        if cum_variances >= min_cum_variances:
            return n_pca

def get_cum_variances(n_pca, data_frame):
    data = data_frame.values
    cov = np.cov(data.T)
    #pdb.set_trace()
    ev, _ = np.linalg.eig(cov)
    cum_variances = np.sort(ev)[-n_pca:].sum() / ev.sum()
    return cum_variances

# 2.3 
n_pca = 3
cum_variances = get_cum_variances(n_pca, features_df)

cum_variances = 0.5
n_pca = get_n_pca(cum_variances, features_df)