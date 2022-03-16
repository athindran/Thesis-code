import numpy as np

names=['rec_auc']
for name in names:
    X = np.genfromtxt('../data/'+name+'.txt', delimiter='\t')
    np.save('../data/'+name, X)
