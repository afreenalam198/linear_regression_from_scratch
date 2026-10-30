# Linear Regression From Scratch: A First-Principles Approach

This repository contains a Jupyter Notebook that builds a complete Linear Regression Machine Learning model entirely from scratch using Python and NumPy. It bridges the gap between pure mathematics and production-level code by implementing iterative optimization, analytical solutions, and matrix algebra without relying on external machine learning frameworks for the core engine.

## Overview

The notebook is structured as a progressive tutorial, starting from fundamental single-feature gradient descent and culminating in a multiple linear regression model evaluated against real-world medical data. Finally, the custom-built engine is benchmarked directly against the industry-standard `scikit-learn` library.

## Table of Contents / Notebook Structure

### 1. Single Feature Gradient Descent
Demonstrates the fundamental mathematics of Machine Learning on a simple toy dataset (predicting house prices based on bedrooms).
* **Concepts:** Hypothesis functions, Mean Squared Error (MSE) loss, and partial derivatives.
* **Implementation:** Functional programming approach to calculate predictions, loss, and parameter updates for a single step of learning.

### 2. The Iterative Training Loop (OOP approach)
Encapsulates the mathematical functions into a robust, object-oriented Python class mimicking the Scikit-Learn API (`fit` and `predict`).
* **Concepts:** Epochs, training loops, and state management (saving weights and biases).
* **Implementation:** Training a model over 1,000 epochs to predict salaries based on years of experience. 

### 3. The Normal Equation (Analytical Solution)
Explores the closed-form mathematical shortcut to finding the global minimum without iterative training. 
* **Concepts:** The Bias Trick (appending a column of 1s), matrix transpositions, and matrix inversion.
* **Math:** $\theta = (X^T X)^{-1} X^T y$
* **Implementation:** A highly efficient, single-line matrix calculation for small-to-medium feature spaces.

### 4. Multiple Linear Regression & Benchmarking (Diabetes Dataset)
The final stage upgrades the model to handle multi-dimensional data using Matrix Multiplication (Dot Products).
* **Concepts:** Vectorized gradients, Z-Score Standardization (Zero-Mean, Unit-Variance) for algorithmic stability, and model evaluation.
* **Implementation:** Training on the Scikit-Learn Diabetes dataset using 4 features (Age, Sex, BMI, BP) and directly comparing the final calculated weights, biases, and MSE against `sklearn.linear_model.LinearRegression`.

## Mathematical Foundations Covered

This notebook explicitly codes the four pillars of Data Science mathematics:
* **Linear Algebra:** Representing datasets as 2D matrices and weights as 1D vectors, optimized via NumPy dot products.
* **Calculus:** Computing partial derivatives of the MSE loss function with respect to weights and bias to determine the steepest descent.
* **Statistics:** Applying Z-score standardization ($X_{\text{scaled}} = \frac{X - \mu}{\sigma}$) to scale raw biological data and prevent gradient explosions.

## Dependencies

To run this notebook, you will need the following Python libraries installed:
* `numpy`: For foundational matrix operations and vectorized math.
* `pandas`: For data manipulation and frame extraction.
* `scikit-learn`: Exclusively used for loading the benchmark datasets (`load_diabetes`) and comparing final metrics (`LinearRegression`, `mean_squared_error`).

## How to Use

1. Clone this repository and open the notebook in Jupyter, Google Colab, or VS Code.
2. Run the cells sequentially to observe the mathematical evolution of the algorithm.
3. Pay special attention to the final benchmark output, which proves that the custom mathematical engine perfectly matches Scikit-Learn's optimized solver down to the sixth decimal place.