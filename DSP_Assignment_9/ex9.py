import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import pdb
'''
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy import linalg as LA
'''

# 2.1  
wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names) 
wine_df = pd.DataFrame(wine_data)
#[178 rows x 14 columns]
#wine_df['Class'].nunique() - 3
features_df = wine_df.drop(['Class'], axis=1)

# 2.2

def PCA(orig_data):
    cov = np.cov(orig_data, rowvar=False)
    eig_vals, eig_vectors = np.linalg.eig(cov)#
    idx = np.argsort(eig_vals)[::-1] # sort eigenvalues in decreasing order
    eig_vals = eig_vals[idx] # re-order eigenvalues
    eig_vectors = eig_vectors[:, idx] # re-order eigenvectors
    pca_data = np.dot(orig_data, eig_vectors)
    return pca_data, eig_vals, eig_vectors

'''
pca = decomposition.PCA(n_components=13)
features_df_std_auto = StandardScaler().fit_transform(features_df)
features_df_pca_auto = pca.fit_transform(features_df_std_auto)
'''
features_df_std_manual = (features_df - features_df.mean()) / features_df.std()
features_df_pca_manual, _, _ = PCA(features_df_std_manual.values)
#print(np.linalg.norm(features_df_pca_auto - features_df_pca_manual))

def get_n_pca(min_cum_variances, eig_vals):
    for n_pca in range(1, len(eig_vals)):
        cum_variances = eig_vals[:n_pca].sum() / eig_vals.sum()
        if cum_variances >= min_cum_variances:
            return n_pca

def get_cum_variances(n_pca, eig_vals):
    cum_variances = eig_vals[:n_pca].sum() / eig_vals.sum()
    return cum_variances

# 2.3 
pca_data, eig_vals, eig_vectors = PCA(features_df_std_manual.values)

n_pca = 3
cum_variances = get_cum_variances(n_pca, eig_vals)
print(cum_variances)
#print('Cumulative sum of explained variance when first {} principal components are selected {}'.format(n_pca, cum_variances))

cum_variances = 0.5
n_pca = get_n_pca(cum_variances, eig_vals)
print(n_pca) #2


# 2.4

fig = plt.figure(figsize=(16, 8))  
plt.scatter(pca_data[:,0], pca_data[:,1], c=wine_df['Class'].values)
plt.title('PCA Projection for Wine dataset')
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Component 2')
plt.show()
plt.savefig('mel_filterbank.jpg') 
