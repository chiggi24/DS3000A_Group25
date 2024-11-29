# Phishing URL Detection with Machine Learning

This repository contains a Python implementation for detecting phishing URLs using various machine learning models. The code evaluates and compares the performance of models like Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and Neural Networks (MLP). The best-performing model is selected based on multiple metrics such as recall, accuracy, precision, and F1 score. The focus is on minimizing false negatives (phishing URLs that are missed) by prioritizing recall in the evaluation process.

## Features

- **Model Comparison**: Compare different classification models based on accuracy, precision, recall, and F1 score.
- **Visualization**: Visualize model performance with scatter plots, confusion matrix heatmaps, and Precision-Recall curves.
- **Feature Importance**: Visualize the importance of features in determining predictions for models that support it (e.g., Logistic Regression, Random Forest).
- **Model Evaluation**: Evaluate the best model on the test set, including performance metrics and visualizations.
- **Grid Search**: Tune hyperparameters using grid search with Stratified K-Fold cross-validation based on recall.
  
## Installation

### Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow` (for neural network models)

You can install the required libraries using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Code Overview

### 1. **Model Definitions**

The code supports five different models for phishing URL detection:
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **Neural Network (MLP)**

Each model is trained and evaluated based on the provided dataset.

### 2. **Evaluation Functions**

- **plot_scatter**: Generates a scatter plot to compare the predicted vs. actual outcomes and their predicted probabilities. Useful for visualizing model performance.
- **print_classification_metrics**: Prints key classification metrics including accuracy, precision, recall, F1 score, and confusion matrix.
- **confusion_matrix_heatmap**: Visualizes the confusion matrix as a heatmap to better understand the model's classification errors.
- **plot_feature_importance**: Displays the importance of each feature for the model’s predictions. Available for tree-based models and linear models (e.g., Logistic Regression).
- **plot_precision_recall_curve**: Plots the Precision-Recall curve and calculates the average precision score, which is particularly useful for imbalanced datasets.
- **evaluate_best_model**: Evaluates the chosen best model on the test data, including a plot of model predictions, confusion matrix, classification metrics, and feature importance.
- **perform_grid_search**: Performs grid search with Stratified K-Fold cross-validation to find the best hyperparameters based on recall score.
- **compare_models**: Compares all models based on accuracy, precision, recall, and F1 score, providing a detailed analysis of each model’s performance.

### 3. **Selecting the Best Model**

The models are evaluated based on recall, with the neural network selected as the best model due to its superior recall score. The selected model also performed well in terms of accuracy and F1 score.

### 4. **Best Model Parameters**

The optimal parameters for the neural network model were:
- **Activation**: 'relu'
- **Alpha**: 0.0001
- **Hidden Layer Sizes**: (10,)
- **Solver**: 'adam'

### 5. **Grid Search and Cross-validation**

A grid search with 5-fold cross-validation is used to fine-tune hyperparameters, optimizing for recall to reduce false negatives. This ensures the best possible model is chosen for phishing URL detection.

## Dataset

The dataset used for this project can be downloaded from the following link:  
[Phishing URL Dataset](https://drive.google.com/file/d/14QqTxF0cXh6lBAeBHfE4pPeFKtuo-Okc/view?usp=sharing)

Make sure to place the dataset in the appropriate directory or adjust the path in the code accordingly.

## Usage

### Step 1: Load Data
Load the dataset of URLs into the program. Ensure that the dataset contains features that can be used to distinguish phishing URLs from legitimate ones.

### Step 2: Train Models
The script will train each of the five models on the dataset.

### Step 3: Model Comparison
Use the `compare_models()` function to compare all models on their performance metrics (accuracy, precision, recall, F1 score).

### Step 4: Evaluate Best Model
After identifying the best model (based on recall), use the `evaluate_best_model()` function to assess its performance on the test data. This includes generating the confusion matrix, feature importance, and Precision-Recall curve.

### Step 5: Model Tuning (Optional)
You can tune the models' hyperparameters using `perform_grid_search()` to further improve the recall score.

## Example Code

```python
# Example: Compare models and evaluate the best model

models = {
    'Logistic Regression': LR_best_model,
    'Random Forest': RF_best_model,
    'Gradient Boosting': GB_best_model,
    'SVM': SVM_best_model,
    'Neural Network (MLP)': MLP_best_model
}

# Compare all models
compare_models(models, X_test, y_test)

# Evaluate the best model
evaluate_best_model(best_model)
```

## Results

The best model for phishing URL detection was the Neural Network (MLP) model, which provided the highest recall, accuracy, and F1 score.
