# Face Recognition with PCA, K-Means, GMM, and Autoencoder

This project tackles a face recognition problem using dimensionality reduction and clustering techniques.
Implementing from scratch PCA, Autoencoders, K-Means, and Gaussian Mixture Models (GMM)

## 📊 Pipeline Overview

### 1. Load and Explore Data

* Load a face dataset
* Display sample images using matplotlib.
* Print the shape of the dataset (images, dimensions).

### 2. Data Splitting

* Randomly split data into:

  * 50% Training
  * 50% Testing

---

## 🔻 PCA

* Implement PCA manually using eigen decomposition of the covariance matrix.
* Tune `alpha` values: `0.8`, `0.85`, `0.9`, `0.95`
* For each `alpha`, retain enough components to match variance threshold.
* Save the eigenvectors
* Reconstruct and visualize sample images after PCA.

#### 🧪 Evaluation Metrics

* Explained Variance ~ 1

---

## 🌀 K-Means Clustering on PCA Data

* Train K-Means with different `k` values: `20`, `40`, `60`
* Apply clustering on PCA-transformed training data
* Predict on validation and test sets

#### 📈 Metrics

* Accuracy: highest at alpha=0.9 & k=60 ~ 0.845 
* Confusion Matrix

---

## 📊 GMM Clustering on PCA Data

* Train GMM on PCA data with `k` values: `20`, `40`, `60`
* Validate and test clustering results

#### 📊 Metrics

* Accuracy: highest at alpha=0.9 & k=60 ~ 0.6700
* F1 Score: highest at alpha=0.9 & k=60 ~ 0.6486
* Confusion Matrix

---

## 🔁 Autoencoder

* Train an autoencoder on image data
* Use encoder output as compressed representation

---

### K-Means on Encoded Data

* Run K-Means with `k = 20, 40, 60`
* Validate and test performance

#### 🧾 Metrics

* Accuracy:  highest at k=60 ~ 0.540
* Confusion Matrix

### GMM on Encoded Data

* Run GMM with `k = 20, 40, 60`
* Validate and test performance

#### 🧾 Metrics

* Accuracy: highest at k=40 ~ 0.5150
* F1 Score: highest at k=40 ~ 0.4802
* Confusion Matrix
