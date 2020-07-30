```python
import numpy as np
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from time import perf_counter
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, trange
# from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch

%matplotlib inline
```

## Computation of Randomized Fourier features Kernels in Parallel

First, let's read in MNIST Data to use as an example, and truncate to the first $1000$ samples.


```python
mnist = pd.read_csv('../datasets/mnist/train.csv')

full_X = mnist[mnist.columns[1:]].values / 255
full_y = mnist.label.values

X = full_X[:1000]
y = full_y[:1000]

n,d = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
```

### Kernel Generation Function

We use random Fourier features to emulate a Gaussian kernel.


```python
def generate_kernel(m=220, s=1/d):
    b = np.random.uniform(low=0, high=2*np.pi, size=(1,m))
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=2*s*np.eye(d), size=m) # m x d
    def ker(x, y):
        z1 = np.cos(x @ W.T + b)
        z2 = np.cos(y @ W.T + b)
        return z1 @ z2.T / m
    return ker
```


```python
parameters = (1/784) * np.arange(0.1,10,0.1)#range(0.1,0.3,0.1)
n_param = parameters.shape[0]
m = 220
```

## Testing parameters using CV: Deterministic Kernel, Series computation

The goal of these experiments is to test a range of parameters values and compare the time it takes for various methods to do that. First, we test each parameter value in series using a deterministic kernel with `sklearn.model_selection.GridSearchCV`.


```python
from sklearn.model_selection import GridSearchCV

start = perf_counter()

params = {'gamma': parameters}
svc = SVC(kernel='rbf')
clf = GridSearchCV(svc, params, cv=3)
clf.fit(X_train, y_train)

print(perf_counter() - start)
print(clf.cv_results_['mean_test_score'])

```

    C:\Users\Sean\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:814: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    

    144.41502749999998
    [0.1275  0.21375 0.34125 0.5375  0.65625 0.725   0.7475  0.76625 0.7825
     0.795   0.79875 0.81    0.815   0.81625 0.82375 0.83    0.8275  0.83125
     0.8375  0.84375 0.84125 0.8425  0.8475  0.84625 0.85    0.84875 0.8475
     0.85    0.85375 0.855   0.85625 0.85625 0.8575  0.8575  0.8575  0.85875
     0.85875 0.85875 0.86    0.85875 0.85875 0.85875 0.86125 0.86125 0.86125
     0.8625  0.8625  0.8625  0.865   0.86625 0.865   0.865   0.86625 0.8675
     0.8675  0.86875 0.87    0.87125 0.87125 0.87125 0.87125 0.87    0.87
     0.87    0.87125 0.87125 0.87375 0.87375 0.875   0.87625 0.87625 0.87625
     0.8775  0.8775  0.8775  0.8775  0.8775  0.8775  0.87875 0.87875 0.87875
     0.87875 0.87875 0.88    0.88    0.88125 0.88125 0.88125 0.88125 0.88125
     0.88125 0.88125 0.88125 0.88125 0.88125 0.88125 0.88125 0.88125 0.88125]
    

## Testing parameters using CV: random Fourier features, Parallel computation

The random Fourier features method has another upshot: since the kernel matrix is approximated by
$$\hat{K} = \frac{1}{m}z(X)z(X)^T$$,
we are effectively approximating the kernel by matrix multiplication. Thus, we may parallelize this code, using _batch matrix multiplication_. We used `torch.bmm` to perform this.


```python
from sklearn.model_selection import StratifiedKFold


# X_train is train_batch x d
# X_test is test_batch x d

n_cv = 3

scores = np.empty((n_cv,n_param))

start = perf_counter()

skf = StratifiedKFold(n_splits = n_cv)
for i,(train_index, test_index) in enumerate(skf.split(X,y)):
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    
    # m x d x n_param
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=2*np.eye(d), size=(n_param, m)).transpose(1,2,0) * np.sqrt(parameters)

    # n_param x m x 1
    b = np.random.uniform(low=0, high=2*np.pi, size=(n_param,m,1))

    # Wtranspose below is n_param x m x d, X_train.T is d x train_batch, their product is n_param x m x train_batch

    placeholder = np.cos(np.dot(W.transpose(2,0,1), X_tr.T) + b) # n_param x m x train_batch

    z11 = torch.from_numpy(placeholder.transpose(0,2,1)) # n_param x train_batch x m
    z2 = torch.from_numpy(placeholder.transpose(0,1,2)) # n_param x m x train_batch

    z12 = torch.from_numpy(np.cos(np.dot(W.transpose(2,0,1), X_te.T) + b).transpose(0,2,1)) # n_param x test_batch x m

    out1 = (1/m) * np.asarray(torch.bmm(z11, z2)) # n_param x train_batch x train_batch
    out2 = (1/m) * np.asarray(torch.bmm(z12, z2)) # n_param x test_batch x train_batch

    for j in range(n_param):
        svm = SVC(kernel='precomputed')
        svm.fit(out1[j], y_tr)
        scores[i,j] = svm.score(out2[j], y_te)

totaltime = perf_counter() - start

print(totaltime)
print(np.mean(scores, axis=0))
```

    44.00260990000001
    [0.12399673 0.14284594 0.18674517 0.31667298 0.44099624 0.49311725
     0.5970826  0.65916469 0.67415371 0.68710705 0.70813439 0.7331066
     0.74706094 0.77212279 0.74411129 0.77707652 0.77698399 0.7740922
     0.7870487  0.78314328 0.78405726 0.78701268 0.78516632 0.79716072
     0.80519897 0.79601248 0.79416928 0.80209423 0.7830326  0.78908645
     0.82207877 0.8090923  0.79708579 0.80504596 0.80205189 0.80605589
     0.81605751 0.81606933 0.81407076 0.80511511 0.8070903  0.81811972
     0.825127   0.81513432 0.81511934 0.82706039 0.8131084  0.82810897
     0.81502653 0.81703774 0.81309946 0.82099074 0.81108852 0.808121
     0.8080495  0.80710528 0.83103126 0.8080708  0.81602149 0.82207588
     0.82909814 0.82299617 0.80809157 0.82110196 0.82997664 0.82008859
     0.81417882 0.82507626 0.81703774 0.81698963 0.82406894 0.82602021
     0.82318467 0.82603493 0.82904425 0.80799271 0.81803613 0.80212709
     0.81811078 0.8369776  0.83606651 0.82513305 0.8309984  0.82111984
     0.82814842 0.82299301 0.82508809 0.82703331 0.82601416 0.82309503
     0.81904552 0.83407922 0.83113932 0.83610858 0.82505839 0.80804977
     0.82310423 0.81110351 0.80809157]
    


```python
randmeans = np.mean(scores,axis=0)
detmeans = clf.cv_results_['mean_test_score']
```


```python
#Error of results
np.linalg.norm(randmeans-detmeans) / np.linalg.norm(detmeans)
```




    0.08477965018786957




```python
#Check where random is maximized, and identify the order of that index in detmeans

random_max_idx = np.argmax(randmeans)

det_val_for_best_rand = detmeans[random_max_idx]

sorted_unique_det = np.unique(np.sort(detmeans))[::-1]

rank = np.where(sorted_unique_det == det_val_for_best_rand)[0][0]

print(f'Best random param value was the {rank}th best det param value')
print(f'Difference in accuracy between best random and best det is {np.max(detmeans)-np.max(randmeans)}')
```

    Best random param value was the 2th best det param value
    Difference in accuracy between best random and best det is 0.044272404147781774
    

## Testing parameters using CV: random Fourier features, Series computation

We compare our previous results to computing the random Fourier features kernels in series, and we note an almost $100\%$ speedup using the parallel method-- this may be attributed to the fact that the machine on which this code was ran had $2$ cores.


```python
# X_train is train_batch x d
# X_test is test_batch x d

n_cv = 3

scores_cv_rnd_np = np.empty((n_cv,n_param))

start = perf_counter()

skf = StratifiedKFold(n_splits = n_cv)
for i,(train_index, test_index) in enumerate(skf.split(X,y)):
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    
    for j,val in enumerate(parameters):
        svm = SVC(kernel=generate_kernel(s=val))
        svm.fit(X_tr, y_tr)
        scores_cv_rnd_np[i,j] = svm.score(X_te, y_te)


totaltime = perf_counter() - start

print(totaltime)
print(np.mean(scores_cv_rnd_np, axis=0))
```

    115.71442639999998
    [0.12399673 0.12399673 0.2126913  0.27203066 0.42601851 0.47274516
     0.58490177 0.57593402 0.65624899 0.70017107 0.71299269 0.73704777
     0.75101736 0.76199262 0.74702546 0.76806173 0.76008027 0.77305491
     0.77408588 0.7910435  0.78508507 0.77108288 0.78392265 0.78911643
     0.8060141  0.80311599 0.80106614 0.802999   0.81012643 0.80005883
     0.79907019 0.81103725 0.80712054 0.80802242 0.8180327  0.81397742
     0.82111063 0.82111352 0.80711395 0.81605778 0.80798405 0.81405262
     0.83013516 0.80510039 0.8319902  0.80510012 0.81612666 0.81593473
     0.82609171 0.81505966 0.81308132 0.81901898 0.83204779 0.81702249
     0.82107172 0.82605307 0.83112407 0.82210269 0.81603096 0.82207588
     0.81308159 0.8151141  0.81803297 0.82602599 0.8191144  0.81008752
     0.83710931 0.8010601  0.82007388 0.83103126 0.80997079 0.82410469
     0.82701543 0.83500268 0.82710508 0.83103415 0.81097125 0.83402271
     0.81914492 0.82609487 0.8219736  0.82012804 0.81612955 0.82608304
     0.82002314 0.81507754 0.82209087 0.82107804 0.82016063 0.82013671
     0.8240629  0.82621765 0.81802692 0.82718263 0.81809842 0.82813921
     0.82014565 0.82412627 0.82402399]
    
