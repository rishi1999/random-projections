# Random Projections
This repository is a collection of our group's work during the Summer@ICERM 2020 REU program.

A link to an Overleaf document with a thorough explanation of the mathematical background and results can be found [here](https://www.overleaf.com/read/ffnmgtrbbkhz).

## Experiments (non-exhaustive)
The `notebooks` folder contains all of our coding experiments.

### Eigenfaces
We use a database of real pictures of faces to extract the components of an average face, which can be added up to reconstruct approximations to any specific face.

##### Eigenfaces calculated deterministically
![Deterministic Eigenfaces](examples/eigenfaces/deterministic_eigenfaces.png)

##### Randomized approximation of the eigenfaces
![Randomized_Eigenfaces](examples/eigenfaces/randomized_eigenfaces.png)

##### Reconstruction of a face using eigenfaces
![Eigenface Reconstruction](examples/eigenfaces/eigenface_reconstruction.png)

### Image Compression
We use randomization to find low-rank approximations to image, making it easier to use these images for data analysis and computation.

##### Example of image compression
![Eigenface Reconstruction](examples/image_compression/image_compression.png)

The left-most image is the original image, and the rest are various forms of approximations.

### JL Lemma
We numerically verify some of the claims made in the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).

### Least Squares
We compute approximate 'least-square' solutions to linear systems that do not have an exact solution, and then compare the accuracy to that of the best of a randomly sampled set of vectors.

## Presentations
The `presentations` folder contains all our files for the biweekly group presentations at ICERM. These mostly consist of Jupyter notebooks with extensive descriptions and explanations of the code / phenomena.

## Installation
To reproduce our results by running the experiments provided in this repository, you will need to have access to all the Python packages imported in the various files. We recommend using Anaconda/conda to set up a virtual environoment and install all packages so as to not interfere with any other projects in your file system.

You will also need to download the datasets used for our experiments (LFW, MNIST). A full list can be found in the linked Overleaf document. A dataset named `dataset1` should be stored in the following directory:

    random-projections/datasets/dataset1

## Authors
Participants: Rishi Advani, Maddy Crim, Sean O'Hagan

TAs: Justin Baker, Liu Yang

Organizers: Akil Narayan (primary), Yanlai Chen (secondary)

## References
A full list of references will be provided in the Overleaf document.
