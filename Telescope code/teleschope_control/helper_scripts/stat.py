import numpy as np

## load a single file ##
X = np.load('../data/rec_TT.npy')
X = X[:, 2:]

print("Number of target acquisition: {} +- {}".format(X[:,0].mean(), X[:,0].std()))
print("Number of end collision: {} +- {}".format(X[:,1].mean(), X[:,1].std()))
print("Number of transient collision: {} +- {}".format(X[:,2].mean(), X[:,2].std()))


## compare two files ##
'''
X = np.load('../data/rec_auc.npy')
Y = np.load('../data/rec_TT.npy')
diff = X-Y

for i in range(5):
    print(diff[:,i].sum())
'''
