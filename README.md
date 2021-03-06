# Random Projections
This repository is a collection of our group's work during the Summer@ICERM 2020 REU program.

- To view our GitHub Pages site, click [here](https://rishi1999.github.io/random-projections/).

- For a more thorough explanation of the mathematical background and results, click [here](./docs/final_report.pdf).

- To access the slides from our final presentation, click [here](./docs/slides.pdf).

### Update!
The code from a continuation of this work can be found in `notebooks/ID_Test.ipynb`. The associated paper can be found [on arXiv](https://arxiv.org/abs/2105.07076).

## Experiments (non-exhaustive)
The `notebooks` folder contains all of our coding experiments.

### Eigenfaces
We use a database of real pictures of faces to extract the components of an average face, which can be added up to reconstruct approximations to any specific face.

##### Eigenfaces calculated deterministically
![Deterministic Eigenfaces](presentations/images/2020-07-10/det_eigenfaces_grid.png)

##### Randomized approximation of the eigenfaces
![Randomized_Eigenfaces](presentations/images/2020-07-10/rand_eigenfaces_grid.png)

##### Reconstruction of a face using eigenfaces
![Eigenface Reconstruction](presentations/images/2020-07-10/reconstructed_image_grid.png)

### Image Compression
We use randomization to find low-rank approximations to image, making it easier to use these images for data analysis and computation.

##### Example of image compression
![Eigenface Reconstruction](examples/image_compression/image_compression.png)

The left-most image is the original image, and the rest are various forms of approximations.

### JL Lemma
We numerically verify some of the claims made in the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).

### Least Squares
We compute approximate 'least-square' solutions to linear systems that do not have an exact solution, and then compare the accuracy to that of the best of a randomly sampled set of vectors.

### Kernel PCA
Certain datasets are not linearly seperable. To solve this problem, we use randomized kernel methods to map data into a higher-dimensional space where PCA is then performed. 

### Kernel SVM
If we want to train a nonlinear classifier on a set of labeled data, one option is to use a Support Vector Machine (SVM). Using a randomized kernel function, we experiment with SVM on the MNIST dataset.

## Presentations
The `presentations` folder contains all our files for the biweekly group presentations at ICERM. These mostly consist of Jupyter notebooks with extensive descriptions and explanations of the code / phenomena.

## Installation
Clone the repository, and install all packages required.

We recommend using Anaconda/conda to set up a virtual environment so as to not interfere with any other projects in your file system. You can run this command to create the environment and install all required packages:

	conda create -n icerm --file package-list.txt

You will also need to download the datasets used for our experiments (e.g., LFW, MNIST). A dataset named `dataset1` should be stored in the following directory:

    random-projections/datasets/dataset1

You can reproduce our results by running the Jupyter notebooks provided.

## Authors
- Rishi Advani
- Madison Crim
- Sean O'Hagan

## References
A full list of references is provided in the final report linked above.

## Acknowledgements
Thank you to our organizers, Akil Narayan and Yanlai Chen, along with our TAs, Justin Baker and Liu Yang, for supporting us throughout this program.
