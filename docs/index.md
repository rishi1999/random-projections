## Summer@ICERM 2020
### Rishi Advani, Maddy Crim, and Sean O'Hagan


<!-- <link rel="shortcut icon" type="image/x-icon" href="favicon.ico/?">

<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png">
<link rel="manifest" href="site.webmanifest"> -->


Hi! We're the random-projections group, and this is our final deliverable for the Summer@ICERM 2020 REU program. Feel free to take a look at our [GitHub repository](https://github.com/rishi1999/random-projections) as well for more content.

<img src="./examples/home/icermmc.gif" alt="ICERM logo" width="200" height="200"/>

### Introduction
In the following experiements we use randomness to compute low-rank matrix approximations and investigate kernel methods.

#### Random Projections
When we have datasets that are too large, traditional methods of calculating low-rank matrix approximations are ineffective. This can be due to factors such as:
- having missing or inaccurate data
- storing all the data simultaneously in memory may not be possible

One solution to these problems is the use of random projections. Instead of directly computing the matrix factorization, we randomly project the matrix onto a lower-dimensional subspace and then compute the factorization. Often, we are able to do this without significant loss of accuracy. With these randomized algorithms, analyzing massive data sets becomes tractable.

#### Kernel Methods
Certain datasets are not always linearally seperable. To solve this problem we can project low-dimensional data into a higher-dimensional 'feature space', such that it is linear separable in the feature space. This enables the model to learn a nonlinear separation of the data.

As before, with large data matrices, computing the kernel matrix can be expensive, so we use randomized methods to approximate the matrix.


### Notebooks
- [Johnson-Lindenstrauss Lemma](./notebooks/html/JL_Lemma.html)
- [Image Compression](./notebooks/html/Image_Compression.html)
- [Eigenfaces](./notebooks/html/Eigenfaces.html)
- [Least Squares](./notebooks/html/Least_Squares.html)
- [Kernel PCA](./notebooks/html/Kernel_PCA.html)
- [Kernel SVM](./notebooks/html/Kernel_SVM.html)
- [Grid Search SVM](./notebooks/html/GridSearchSVM.html)

### And more...

To understand the theory behind the experiments with more mathematical rigor, click [here](./final_report.pdf) to view our final report.

To access the slides from our final presentation, click [here](./slides.pdf).

#### Acknowledgments

Thank you to ICERM for (virtually) hosting us this summer, and thank you to all the staff for making this program possible. Thank you to our organizers, Akil Narayan and Yanlai Chen, along with our TAs, Justin Baker and Liu Yang, for supporting us throughout this program.

The ICERM logo GIF was generated with the use of [imcmc](https://github.com/ColCarroll/imcmc).
