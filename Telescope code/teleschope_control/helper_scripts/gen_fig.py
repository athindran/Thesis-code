import numpy as np
import matplotlib.pyplot as plt

## 1. generate full plots for a single algorithm
X = np.genfromtxt('../data/eps_auc.txt',delimiter='\t')
X = X[1:, :]

# rescale nCompTar
X[:, 1:3] = X[:, 1:3]/100.0

label = ['# of Target Acquisition / 100', '# of End Collision', '# of Transient Collision']
for i in range(3):
    plt.plot(X[:,0], X[:, 1+i*2], linewidth=1, label=label[i])
    plt.fill_between(X[:,0], X[:,1+i*2]-X[:,(i+1)*2], X[:,1+i*2]+X[:,(i+1)*2], alpha=0.5)
plt.xlabel(r'$\epsilon$')
plt.ylabel('counts')
plt.legend()
plt.grid()
plt.savefig('../figs/eps_effect.pdf')


## 2. compare default (greedy) with auction in eps plot
X = np.genfromtxt('../data/eps_def.txt',delimiter='\t')
X = X[1:, :]
Y = np.genfromtxt('../data/eps_auc.txt',delimiter='\t')
Y = Y[1:, :]
plt.figure()
plt.plot(Y[:,0], Y[:, 1], linewidth=1, label='auction')
plt.fill_between(Y[:,0], Y[:,1]-Y[:,2], Y[:,1]+Y[:,2], alpha=0.5)
plt.plot(X[:,0], X[:, 1], linewidth=1, label='default')
plt.fill_between(X[:,0], X[:,1]-X[:,2], X[:,1]+X[:,2], alpha=0.5)
plt.xlabel(r'$\epsilon$')
plt.ylabel('# of Target Acquisition')
plt.legend()
plt.grid()
plt.savefig('../figs/compare_eps.pdf')

## 3. compare default (greedy) with auction in counts diff plot
X = np.load('../data/rec_def.npy')
X = X[:, 2]
Y = np.load('../data/rec_auc.npy')
Y = Y[:, 2]

val, freq = np.unique(Y-X, return_counts=True)
freq = freq/len(X)

plt.figure()
plt.scatter(val, freq)

# mean plot
x = np.ones(100)*0.7
y = np.linspace(0,0.45,num=100)
plt.plot(x,y, color='red', linewidth=1, label='mean')

plt.xlabel('# from auction - # from greedy')
plt.ylabel('probability')
plt.grid()
plt.legend()
plt.savefig('../figs/compare_freq.pdf')
