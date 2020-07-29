```python
import numpy as np
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from time import perf_counter
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

%matplotlib inline
```

# SVM Classification using Random Fourier Features Kernel

First, we read in MNIST data and create two sets of training and testing data, containing $1000$ and $10,000$ samples, respectively. These will be used for various experiments


```python
mnist = pd.read_csv('../datasets/mnist/train.csv')
# mnist = pd.read_csv('../datasets/fashion/fashion-mnist_test.csv')

full_X = mnist[mnist.columns[1:]].values / 255
full_y = mnist.label.values

X = full_X[:1000]
y = full_y[:1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

n,d = X.shape


large_X = full_X[:10000]
large_y = full_y[:10000]
large_X_train, large_X_test, large_y_train, large_y_test = train_test_split(large_X, large_y, test_size=0.2, random_state=15)
```


```python
def generate_kernel(m=350, s=1/d):
    b = np.random.uniform(low=0, high=2*np.pi, size=(1,m))
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=2*s*np.eye(d), size=m) # m x d
    def ker(x, y):
        z1 = np.cos(x @ W.T + b)
        z2 = np.cos(y @ W.T + b)
        return z1 @ z2.T / m
    return ker
```

#### Time comparison on $10,000$ MNIST images

First, we compare the time needed to train and test a SVM on $10,000$ MNIST images ($784$ features)


```python
start = perf_counter()
svm = SVC(gamma='auto')
svm.fit(large_X_train, large_y_train)
det_score = svm.score(large_X_test, large_y_test)
end = perf_counter()
det_time = end - start


iterations = 100
scores = np.empty(iterations)
times = np.empty(iterations)

for i in tqdm(range(iterations)):
    start = perf_counter()
    random_svm = SVC(kernel=generate_kernel())
    random_svm.fit(large_X_train, large_y_train)
    end = perf_counter()
    times[i] = end - start
    scores[i] = random_svm.score(large_X_test, large_y_test)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [04:39<00:00,  2.79s/it]
    


```python
print(f'Gaussian kernel: Accuracy: {det_score}, time: {det_time}')
print(f'Random kernel: Mean accuracy: {np.mean(scores)}, stdev: {np.std(scores)}, mean time: {np.mean(times)}')
print(f'Acc. stats: Q0: {np.min(scores)}, Q1: {np.quantile(scores, 0.25)}, Q2: {np.median(scores)}, Q3: {np.quantile(scores,0.75)}, Q4: {np.max(scores)}')
```

    Gaussian kernel: Accuracy: 0.9195, time: 37.9916527
    Random kernel: Mean accuracy: 0.8909999999999999, stdev: 0.004221966366516911, mean time: 2.1432035219999994
    Acc. stats: Q0: 0.881, Q1: 0.887875, Q2: 0.8915, Q3: 0.8945, Q4: 0.9005
    

We observe that the random Fourier features kernel comes close to the deterministic Gaussian kernel in terms of accuracy ($96.75\%$ as accurate as deterministic), but takes only about $5.9\%$ as long. Considering the impressively low standard deviation makes this approach very useful for large datasets.

## Accuracy as a function of gamma

For the remainder of the experiments, we switch to a subset of MNIST containing only $1000$ samples. This is because the large dataset is useful for timing comparisons, but makes many of the following experiments computationally intractable.

First, we example the effect of varying gamma on the accuracy of the SVM, using both a deterministic and random kernel.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
domain = (1/784) * np.asarray([.125,.25,.5,1,2,4,8,16,32,64,128])
```

### Deterministic Gaussian kernel


```python
scores = []

for val in tqdm(domain):
    svm = SVC(gamma=val)
    svm.fit(X_train, y_train)
    scores.append(svm.score(X_test, y_test))
    
plt.plot(domain, scores)
plt.xlabel('$\gamma$')
plt.ylabel('Accuracy')
plt.title('Deterministic Gaussian kernel SVM')
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 11/11 [00:11<00:00,  1.02s/it]
    




    Text(0.5, 1.0, 'Deterministic Gaussian kernel SVM')




![png](output_11_2.png)


### Random Fourier features approximation of Gaussian kernel


```python
iterations = 100
scores = np.empty((domain.shape[0], iterations))
    
for i,val in enumerate(domain):
    for j in range(iterations):
        random_svm = SVC(kernel=generate_kernel(s=val))
        random_svm.fit(X_train, y_train)
        scores[i,j] = random_svm.score(X_test, y_test)

stat = np.empty((7, domain.shape[0]))
stat[0] = np.min(scores, axis=1)
stat[1] = np.quantile(scores, 0.25, axis=1)
stat[2] = np.median(scores, axis=1)
stat[3] = np.quantile(scores, 0.75, axis=1)
stat[4] = np.max(scores, axis=1)
stat[5] = np.mean(scores, axis=1)
stat[6] = np.std(scores, axis=1)
```


```python
plt.plot(domain, stat[2], c='b', label='Median')
plt.fill_between(domain, stat[4], stat[0], alpha=0.2, label='Range', lw=1)
plt.fill_between(domain, stat[1], stat[3], alpha=0.5, label='IQR', lw=1)
plt.xlabel('$\gamma$')
plt.ylabel('Accuracy')
#plt.set_yscale('log')
plt.legend(loc='upper right')
plt.title('Randomized Kernel SVM accuracy \'continuous boxplot\'')
```




    Text(0.5, 1.0, "Randomized Kernel SVM accuracy 'continuous boxplot'")




![png](output_14_1.png)


In the randomized case, we sample $\mathbf{w}_i \sim N(0,2\gamma\mathbf{I})$ to approximate a deterministic rbf kernel with parameter $\gamma$.

### Accuracy as a function of $m$ (number of random Fourier features sampled)


```python
m_domain = np.asarray([10, 50, 100, 200, 400, 750])
np.arange(start=100, stop=800, step=100)
```




    array([100, 200, 300, 400, 500, 600, 700])




```python
iterations = 100
scores = np.empty((m_domain.shape[0], iterations))
times = np.empty((m_domain.shape[0], iterations))
    
for i,val in enumerate(m_domain):
    for j in range(iterations):
        st = perf_counter()
        random_svm = SVC(kernel=generate_kernel(m=val))
        random_svm.fit(X_train, y_train)
        scores[i,j] = random_svm.score(X_test, y_test)
        times[i,j] = perf_counter() - st
        
stat = np.empty((7, m_domain.shape[0]))
stat[0] = np.min(scores, axis=1)
stat[1] = np.quantile(scores, 0.25, axis=1)
stat[2] = np.median(scores, axis=1)
stat[3] = np.quantile(scores, 0.75, axis=1)
stat[4] = np.max(scores, axis=1)
stat[5] = np.mean(scores, axis=1)
stat[6] = np.std(scores, axis=1)

meantimes = np.mean(times, axis=1)
```


```python
st = perf_counter()
det_svm = SVC(gamma='auto')
det_svm.fit(X_train, y_train)
det_svm.score(X_test, y_test)
det_time = perf_counter() - st

relativemeantimes = meantimes / det_time
```


```python
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(m_domain, stat[2], c='b', label='Median accuracy')
ax1.fill_between(m_domain, stat[4], stat[0], alpha=0.2, label='Accuracy range', lw=1)
ax1.fill_between(m_domain, stat[1], stat[3], alpha=0.5, label='Accuracy IQR', lw=1)
ax1.set_xlabel('m')
ax1.set_ylabel('Accuracy')
#plt.set_yscale('log')

ax2.plot(m_domain, relativemeantimes, c='r', label='Time')
ax2.set_ylabel('Time Relative to Deterministic')
ax2.set_ylim(0.2,0.5)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

plt.title('Randomized Kernel SVM accuracy as a function of m')
```




    Text(0.5, 1.0, 'Randomized Kernel SVM accuracy as a function of m')




![png](output_20_1.png)



```python
plt.plot(m_domain, stat[2], c='b', label='Median')
plt.fill_between(m_domain, stat[4], stat[0], alpha=0.2, label='Range', lw=1)
plt.fill_between(m_domain, stat[1], stat[3], alpha=0.5, label='IQR', lw=1)
plt.xlabel('m')
plt.ylabel('Accuracy')
#plt.set_yscale('log')
plt.legend(loc='upper right')
plt.title('Randomized Kernel SVM accuracy as a function of m')
```




    Text(0.5, 1.0, 'Randomized Kernel SVM accuracy as a function of m')




![png](output_21_1.png)


## Randomly Approximating the Kernel over a range of parameters

Refer to section $4.4$ for documentation of this method. The goal is to approximate a kernel as a random sample average over a range of parameters, useful for when the optimal parameter value is unknown but an approximate range is known. In this case, we use a $sinc$ kernel as the underlying parametric family being approximated. Letting $B$ denote the parameter of the $sinc$ kernel, we sampled $B$ uniformly from the range $(0.5,4)$ for the following experiments.


```python
def generate_kernel(m=100, q=10, s=1/d, scale=True):
    q_arr = np.random.uniform(low=0.5, high=4, size=q) 
    if scale:
        s = s/np.var(X_train)
    std = np.sqrt(2*s)
    W = np.random.uniform(low=-std*q_arr, high=std*q_arr, size=(m,d,q))
    b = np.random.uniform(low=0, high=2*np.pi, size=(1,m,1))
    
    def ker(x, y):
        z1 = np.cos(np.dot(x, W) + b)
        z2 = np.cos(np.dot(y, W) + b)
        res = np.tensordot(z1, z2, axes=([1,2],[1,2])) / (m*q)
        return res
    
    return ker
```

    x: test, d
    y: train, d
    W: q, m, d ---- m, d, q
    b: 1, m
    z1: test, m ---- test, m, q
    z2: train, m ---- train, m, q
    res: test, train

## Experiments

### Deterministic Gaussian Kernel with Default $\gamma$


```python
gamma_val = 'auto'
```


```python
det_svm = SVC(gamma=gamma_val)
det_svm.fit(X_train, y_train)
det_score = det_svm.score(X_test, y_test)
```


```python
iterations = 100
det_times = np.empty((iterations))

for i in trange(iterations):
    start = perf_counter()
    det_svm = SVC(gamma=gamma_val)
    det_svm.fit(X_train, y_train)
    _ = det_svm.score(X_test, y_test)
    det_times[i] = perf_counter() - start
```

    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [01:27<00:00,  1.15it/s]
    


```python
print(f'Accuracy: {det_score}, time: {np.mean(det_times)}')
```

    Accuracy: 0.855, time: 0.8716985479999971
    

### Randomized Averaged $sinc$ Kernel


```python
iterations = 100
scores = np.empty((iterations))
times = np.empty((iterations))
    

for i in trange(iterations):
    start = perf_counter()
    random_svm = SVC(kernel=generate_kernel())
    random_svm.fit(X_train, y_train)
    scores[i] = random_svm.score(X_test, y_test)
    times[i]= perf_counter() - start

stat = np.empty(7)
stat[0] = np.min(scores)
stat[1] = np.quantile(scores, 0.25)
stat[2] = np.median(scores)
stat[3] = np.quantile(scores, 0.75)
stat[4] = np.max(scores)
stat[5] = np.mean(scores)
stat[6] = np.std(scores)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [04:53<00:00,  2.94s/it]
    


```python
print(f'Mean accuracy: {stat[5]}, stdev: {stat[6]}, mean time: {np.mean(times)}')
print(f'Acc. stats: Q0: {stat[0]}, Q1: {stat[1]}, Q2: {stat[2]}, Q3: {stat[3]}, Q4: {stat[4]}')
```

    Mean accuracy: 0.8524, stdev: 0.022388389848312006, mean time: 2.9366895539999973
    Acc. stats: Q0: 0.78, Q1: 0.83875, Q2: 0.855, Q3: 0.865, Q4: 0.895
    

#### Histogram of Accuracies


```python
plt.hist(scores, bins=30);
```


![png](output_35_0.png)


#### Histogram of Times


```python
plt.hist(times, bins=40);
```


![png](output_37_0.png)


#### 2D Histogram of Accuracies, Times


```python
plt.hist2d(scores, times, bins=10, cmap='inferno');
```


![png](output_39_0.png)


### 'Cheating' Randomized Method (Training on the Test Data)

The purpose of this experiment is to see whether the model is able to recover all of the modes that it learns on.


```python
iterations = 100
cscores = np.empty((iterations))
ctimes = np.empty((iterations))
    

for i in trange(iterations):
    start = perf_counter()
    crandom_svm = SVC(kernel=generate_kernel())
    crandom_svm.fit(X_test, y_test) # caution: do not blindly copy this code.
                                   # training on the test data is generally frowned upon.
    cscores[i] = crandom_svm.score(X_test, y_test)
    ctimes[i]= perf_counter() - start

cstat = np.empty(7)
cstat[0] = np.min(cscores)
cstat[1] = np.quantile(cscores, 0.25)
cstat[2] = np.median(cscores)
cstat[3] = np.quantile(cscores, 0.75)
cstat[4] = np.max(cscores)
cstat[5] = np.mean(cscores)
cstat[6] = np.std(cscores)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [01:39<00:00,  1.01it/s]
    


```python
print(f'Mean accuracy: {cstat[5]}, stdev: {cstat[6]}, mean time: {np.mean(ctimes)}')
print(f'Acc. stats: Q0: {cstat[0]}, Q1: {cstat[1]}, Q2: {cstat[2]}, Q3: {cstat[3]}, Q4: {cstat[4]}')
```

    Mean accuracy: 0.8471999999999998, stdev: 0.03527265229607777, mean time: 0.9924936850000017
    Acc. stats: Q0: 0.73, Q1: 0.83, Q2: 0.845, Q3: 0.87125, Q4: 0.92
    


```python
plt.hist(cscores, bins=20);
```


![png](output_44_0.png)



```python
plt.hist(ctimes, bins=30);
```


![png](output_45_0.png)



```python
plt.hist2d(cscores, ctimes, bins=10, cmap='inferno');
```


![png](output_46_0.png)

