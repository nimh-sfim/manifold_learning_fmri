# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: opentsne
#     language: python
#     name: opentsne
# ---

# # Figure 4 - TSNE Step-by-step
#
# This notebook contains code available at Laurens Van der Maaten T-SNE home page (https://lvdmaaten.github.io/tsne/) 

# + [markdown] tags=[]
# # 1. Load tvFC Data
# -

from openTSNE import TSNE
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
print(sklearn.__version__)

from utils.basics import load_representative_tvFC_data
import pandas as pd
import hvplot.pandas
from utils.basics import task_cmap_caps as cmap
X_df = load_representative_tvFC_data()

labels  = pd.Series(X_df.columns)
X       = X_df.values.T
(n_wins, n_conns)  = X.shape         # n = number of samples | d = original number of dimensions
print('++ INFO: Input data shape = %s' % str(X.shape))

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=.33, random_state=42)
print("%d training samples" % x_train.shape[0])
print("%d test samples" % x_test.shape[0])

tsne = TSNE(n_jobs=30, perplexity=100, verbose=True, metric='euclidean')

embedding_train = tsne.fit(x_train)

aux = pd.DataFrame(embedding_train,columns=['x','y'])
aux['labels'] = y_train.values
aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

# %time embedding_test = embedding_train.transform(x_test)

aux2 = pd.DataFrame(embedding_test,columns=['x','y'])
aux2['labels'] = y_test.values
aux2.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square', alpha=0.1) * aux2.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

# ***

from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

# %%time
affinities_train = affinity.PerplexityBasedNN(
    x_train,
    perplexity=30,
    metric="euclidean",
    n_jobs=24,
    random_state=42,
    verbose=True,
)

# %time init_train = initialization.pca(x_train, random_state=42)

embedding_train = TSNEEmbedding(
    init_train,
    affinities_train,
    negative_gradient_method="fft",
    n_jobs=8,
    verbose=True,
)

# %time embedding_train_1 = embedding_train.optimize(n_iter=250, exaggeration=20, momentum=0.5)

aux = pd.DataFrame(embedding_train_1,columns=['x','y'])
aux['labels'] = y_train.values
aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

# %time embedding_train_2 = embedding_train_1.optimize(n_iter=500, momentum=0.8)

aux = pd.DataFrame(embedding_train_2,columns=['x','y'])
aux['labels'] = y_train.values
aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

# ***

import openTSNE

# %%time
embedding_annealing = openTSNE.TSNE(
    perplexity=500, metric="cosine", initialization="pca", n_jobs=8, random_state=3
).fit(X)

aux = pd.DataFrame(embedding_annealing,columns=['x','y'])
aux['labels'] =labels
aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

# %time embedding_annealing.affinities.set_perplexity(50)

# %time embedding_annealing = embedding_annealing.optimize(250, momentum=0.8)

aux = pd.DataFrame(embedding_annealing,columns=['x','y'])
aux['labels'] =labels
aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')

# ***

# %%time
affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    X,
    perplexities=[50,100, 200],
    metric="cosine",
    n_jobs=24,
    random_state=43,
)

# %time init = openTSNE.initialization.pca(X, random_state=43)

embedding_multiscale = openTSNE.TSNE(n_jobs=24).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

aux = pd.DataFrame(embedding_multiscale,columns=['x','y'])
aux['labels'] =labels
aux.hvplot.scatter(x='x',y='y',c='labels', cmap=cmap, aspect='square')


