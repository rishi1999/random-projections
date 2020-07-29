# Eigenfaces

In this notebook, we apply the deterministic and randomized variants of the SVD to a large, real-life dataset. We will be using the [Labeled Faces in the Wild (LFW) dataset](http://vis-www.cs.umass.edu/lfw/), which consists of ~$8000$ RGB images of dimensions $250\times 250$. It's computationally infeasible for us to use the whole dataset, so we load a subset (1000 images).

#### Import packages


```python
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob
from time import perf_counter
from tqdm import tqdm, trange

%matplotlib inline
```


```python
%cd ..
from rputil import *
%cd -
```

    C:\Users\Sean\random-projections
    C:\Users\Sean\random-projections\notebooks
    

#### Load data

We flatten each image to represent it as vector of length $250\cdot 250 \cdot 3 = 187500$. This will yield a data matrix $A$ of size $187500 \times 1000$.


```python
m = 187500
n = 1000

data = np.empty((m, n))

for i,filename in enumerate(glob.iglob('../datasets/lfw/**/*.jpg', recursive = True)):
    if i >= n:
        break
    im=Image.open(filename)
    data[:,i] = np.asarray(im).flatten()
    im.close()
```


```python
data /= 255 # convert from int to float representation
```

#### Column-center the data

This allows us to interpret the data in terms of the way it deviates from an 'average' face.


```python
data_mean = np.expand_dims(np.mean(data, axis=1), 1)
data -= data_mean
```

#### Calculate eigenfaces

We compute the singular value decomposition,
$$A = U \Sigma V^* \,,$$
and then truncate $U$, keeping only the first $k$ columns, to use it as a basis for a rank-$k$ approximation.

Similarly, we use random projection to compute a randomized approximation of the SVD.


```python
start = perf_counter()
det_U, det_Sigma, det_Vh = np.linalg.svd(data, full_matrices=False)
end = perf_counter()
det_time = end - start
```


```python
k = 350
```


```python
det_basis = det_U[:,:k]
print(det_basis.shape)
```

    (187500, 350)
    


```python
rand_basis = random_svd_rank_k(data, k, power=0, only_basis=True)
print(rand_basis.shape)
```

    (187500, 350)
    


```python
def normalize_basis(basis):
    return (basis - np.min(basis))/(np.max(basis)-np.min(basis))
```


```python
normalized_det_basis = normalize_basis(det_basis)
```


```python
normalized_rand_basis = normalize_basis(rand_basis)
```

Let's take a look at some of the first few faces in the dataset.

Some things to note:
- there can be multiple pictures of the same person
- the faces are all mostly centered and facing straight, but there are some exceptions


```python
def plot_examples(rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 / cols * rows))
    for i,ax in enumerate(axes.flat):
        ax.imshow((data[:,i]+np.squeeze(data_mean)).reshape(250,250,3))
```


```python
plot_examples(2, 5)
```


![png](output_23_0.png)


And now the eigenvectors, or 'eigenfaces,' comprising a basis for the image space:


```python
def plot_eigenfaces(basis, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(15, 15 / cols * rows))
    for i in range(rows):
        for j in range(cols):
            ax[i][j].imshow(basis[:,i * cols + j].reshape(250,250,3))
```


```python
plot_eigenfaces(normalized_det_basis, 2, 5)
```


![png](output_26_0.png)



```python
plot_eigenfaces(normalized_rand_basis, 2, 5)
```


![png](output_27_0.png)


### Reconstruct an image 

Now that we have a basis, we may represent arbitrary vectors in our image space as linear combinations of basis vectors. To do this, we solve the linear system
$$A\vec{x}=\vec{b}-m \,,$$
where $A$ denotes the matrix of which our basis vectors are columns, $\vec{b}$ denotes our specific image, and $m$ denotes the (column) mean of the dataset. Since we know our basis is orthonormal, we can use a shortcut, computing the coefficients $\vec{x}$ as 
$$\vec{x} = (\vec{b}-m)A \,.$$

The reason why this technique is useful is that any arbitrary image that is similar to those in the dataset may be represented well using this basis.


```python
specific_image = np.asarray(Image.open('../examples/eigenfaces/sean250.png')).flatten() / 255
```


```python
det_coefficients = (specific_image - np.squeeze(data_mean)) @ det_basis
```


```python
rand_coefficients = (specific_image - np.squeeze(data_mean)) @ rand_basis
```


```python
det_reconstructed_image = np.squeeze(data_mean) + (det_basis @ det_coefficients)
```


```python
rand_reconstructed_image = np.squeeze(data_mean) + (rand_basis @ rand_coefficients)
```

Here is an example of a face reconstructed using the eigenfaces:


```python
fig, ax = plt.subplots(1, 3, figsize=(15,10))
ax[0].imshow(specific_image.reshape(250,250,3))
ax[0].set_xlabel('Original Image')
ax[1].imshow(det_reconstructed_image.reshape(250,250,3))
ax[1].set_xlabel(f'Rank-{k} reconstruction using SVD basis')
ax[2].imshow(rand_reconstructed_image.reshape(250,250,3))
ax[2].set_xlabel(f'Rank-{k} reconstruction using RSVD basis');
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


![png](output_36_1.png)


These animations show the quality of the image representation as we increase the rank of the matrix approximation:

Aaron Eckhart | Sean | President Obama | Ellie Goulding
- | - | - | -
![Guy](../examples/eigenfaces/guy250.png) | ![Sean](../examples/eigenfaces/sean250.png) | ![Pres](../examples/eigenfaces/president_obama250.png) | ![Ellie](../examples/eigenfaces/ellie250.png)
![Anim](../presentations/images/2020-07-10/anim/guy.gif) | ![Anim](../presentations/images/2020-07-10/anim/sean.gif) | ![Anim](../presentations/images/2020-07-10/anim/pres.gif) | ![Anim](../presentations/images/2020-07-10/anim/ellie.gif)

## Statistical Analysis

We have seen visually that we are able to compute high-quality approximations to faces, whether they are in the dataset or not. Now, we run several experiments to back up our visual intuition with hard numbers.

### Relative error of RSVD compared to SVD

We compute approximations for varying ranks $k$, and evaluate the relative error for each rank.


```python
domain = np.arange(50, 550, step=50)

rand_times = []
errors = []

for k in tqdm(domain):
    det_approx = svd_rank_k(data, k)
    
    start = perf_counter()
    rand_approx = random_svd_rank_k(data, k, power=0)
    end = perf_counter()
    rand_time = end - start
    
    rand_times.append(rand_time / det_time)
    
    
    error = (np.linalg.norm(data - rand_approx) 
             - np.linalg.norm(data - det_approx)) / np.linalg.norm(data - det_approx)
    errors.append(error)
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [15:50<00:00, 95.01s/it]
    


```python
fig, ax = plt.subplots()

ax.set_xlabel('approximation rank')

ax.plot(domain, errors, c='r', label='Error')
ax.set_ylabel('relative error')

ax2 = ax.twinx()

ax2.plot(domain, rand_times, c='b', label='Time')
ax2.set_ylabel('relative time elapsed')

fig.legend(loc='upper right');
```


![png](output_44_0.png)


### RSVD/SVD error relative to original images

Here we compare the quality of the RSVD and SVD approximations to the original images.


```python
n_sample = 10
rsvd_error = np.empty((domain.shape[0], n_sample))
svd_error = np.empty(domain.shape[0])

rsvd_stat = np.empty((7,domain.shape[0]))

for i in trange(domain.shape[0]):
    svd_error[i] = np.linalg.norm((det_U[:,:domain[i]] 
                                   @ np.diag(det_Sigma[:domain[i]]) 
                                   @ det_Vh[:domain[i]]) 
                                  - data) / np.linalg.norm(data)
    for j in range(n_sample):
        rsvd_error[i][j] = np.linalg.norm(random_svd_rank_k(data, domain[i], power=0) 
                                          - data) / np.linalg.norm(data)
```

    100%|█████████████████████████████████████████████████████████████████████████████████| 10/10 [28:51<00:00, 173.16s/it]
    


```python
for i in range(domain.shape[0]):
    rsvd_stat[0][i] = np.min(rsvd_error[i])
    rsvd_stat[1][i] = np.quantile(rsvd_error[i], 0.25)
    rsvd_stat[2][i] = np.median(rsvd_error[i])
    rsvd_stat[3][i] = np.quantile(rsvd_error[i], 0.75)
    rsvd_stat[4][i] = np.max(rsvd_error[i])
    rsvd_stat[5][i] = np.mean(rsvd_error[i])
    rsvd_stat[6][i] = np.std(rsvd_error[i])
```


```python
fig, ax = plt.subplots(1,2, figsize=(20,5))

ax[0].plot(domain, svd_error, c='b', label='SVD')
ax[0].plot(domain, rsvd_stat[5], c='c', label='mean RSVD')
ax[0].set_xlabel('k')
ax[0].set_ylabel('Relative Error')
# ax[0].set_yscale('log')
ax[0].legend(loc='upper right')
ax[0].set_title('Error')

ax[1].plot(domain, rsvd_stat[2], c='b', label='Median')
ax[1].fill_between(domain, rsvd_stat[4], rsvd_stat[0], alpha=0.2, label='Range', lw=1)
ax[1].fill_between(domain, rsvd_stat[1], rsvd_stat[3], alpha=0.5, label='IQR', lw=1)
ax[1].set_xlabel('k')
ax[1].set_ylabel('Relative Error, log scale')
# ax[1].set_yscale('log')
ax[1].legend(loc='upper right')
ax[1].set_title('Randomized SVD \'continuous boxplot\'')


plt.show()
```


![png](output_49_0.png)



```python
print(f"For k=200: Mean={rsvd_stat[5][3]}, StDev={rsvd_stat[6][3]}")
```

    For k=200: Mean=0.42221543583274085, StDev=0.0006367842431655264
    

While the relative accuracy for both SVD and RSVD isn't perfect for particularly low-rank approximations, it is still reasonable. With more computing power, we could run the decompositions on more data and potentially get better results.

Another thing to note is the high precision of the RSVD approximation (due to the concentration of measure phenomenon).
