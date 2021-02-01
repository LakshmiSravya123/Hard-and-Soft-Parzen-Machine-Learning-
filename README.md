# Prazen (Hard and Soft)

Hard and Soft Parzen in machine learning without using SKlearn and built-in libraries 

## Installation
Download the parzen.py

## Dataset
iris.txt

## Import
import numpy as np

iris = np.loadtxt('iris.txt')

## Funtions

Q1.feature_means: An array containing the empirical means of
each feature, from all examples present in the dataset. Make sure
to maintain the original order of features.

Q1.covariance_matrix: A  4 × 4 matrix that represents the empirical
the covariance matrix of features on the whole dataset.

Q1.feature_means_class_1: An array containing the empirical
means of each feature, but only from examples in class 1. The
possible classes in the iris dataset are 1, 2, and 3.

Q1.covariance_matrix_class_1 : A 4 × 4 matrix that represents
the empirical covariance matrix of features, but only from examples
in-class 1.

## To do 

1) Implement Parzen with hard window parameter h.

f = HardParzen(h) initiates the algorithm with parameter h
f.train(X, Y) trains the algorithm, where X is a n × m matrix of n
training samples with m features, and Y is an array containing the n
labels. The labels are denoted by integers, but the number of classes
in Y can vary.


f.compute_predictions(X_test) computes the predicted labels and
returns them as an array of the same size as X_test. X_test is a k × m
matrix of k test samples with the same number of features as X. This
function is called only after training on (X, Y ).

2) Implement Parzen with a soft window.
We will use a radial basis function (RBF) kernel (also known as Gaussian kernel)


3) Implement a function split_dataset that splits the Iris
dataset as follows:

A training set consisting of the samples of the dataset with indices
which have a remainder of either 0, 1, or 2 when divided by 5

A validation set consisting of the samples of the dataset with
indices which have a remainder of 3 when divided by 5.

A test set consisting of the samples of the dataset with indices
which have a remainder of 4 when divided by 5.



## Language used
Python

