# Predicting User Churn in Music Streaming Service

## Project Overview

This project focuses on predicting user churn for a music streaming service based on their activity data. The goal is to develop a machine learning model that can predict whether a user will churn (leave the service) based on features such as the number of songs played, session duration, and subscription activity.

## Motivation

Customer churn is a critical issue for subscription-based services, and predicting churn allows businesses to take proactive steps to retain valuable users. This project demonstrates how machine learning models can be applied to predict churn, providing actionable insights for service providers.

## Libraries and Tools Used

- **PySpark**: Used for distributed data processing and machine learning tasks.
- **Pandas**: For data manipulation and exploration.
- **Matplotlib**: For creating visualizations to analyze the data.
- **Scikit-learn**: For model evaluation and metrics.
- **Jupyter Notebook**: Used for documenting and running the analysis interactively.
- **Python 3.x**: The programming language used for the implementation.

## Files in the Repository

### 1. **README.md**
   - This file, providing an overview of the project, setup instructions, and results.

### 2. **Sparkify.ipynb**
   - A Jupyter Notebook detailing the data preprocessing, model training, evaluation, and results.

### 3. *mini_sparkify_event_data.json**
   - Contains the dataset used in the project. This directory includes any raw or preprocessed data files used during analysis.

### 4. **requirements.txt**
   - Lists all dependencies required to run the project, such as `pandas`, `matplotlib`, `pyspark`, etc.

## Project Definition

This project aims to build a model that can predict whether a user of a music streaming service will churn based on their activity. The features used include the number of actions performed, the number of songs played, average session time, and subscription activity. The project uses PySpark to train and evaluate three machine learning models: Logistic Regression, Random Forest, and Decision Tree.

## Analysis

The project follows a structured approach:

1. **Data Preprocessing**: Missing values were handled, and necessary features were engineered. A `VectorAssembler` was used to assemble the features into a single vector for model training.
   
2. **Model Training**: Three models were trained on the dataset:
   - Logistic Regression
   - Random Forest Classifier
   - Decision Tree Classifier
   
3. **Model Evaluation**: Models were evaluated based on metrics such as F1 score and AUC to determine their performance.

4. **Results**: The models' performances were compared, and the best model was selected for final evaluation.

## Results

- **Best Model**: Random Forest Classifier achieved the highest F1 score on the test set.
- **Model Performance**:
  - **Logistic Regression**: F1 Score = 0.72
  - **Random Forest**: F1 Score = 0.85
  - **Decision Tree**: F1 Score = 0.78

## Conclusion

The Random Forest Classifier outperformed the other models and is recommended for predicting user churn. By using this model, businesses can identify users likely to churn and take actions to retain them, improving overall customer retention.

## How to Run the Project

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- PySpark
- Jupyter Notebook (for running the analysis interactively)

### Setup Instructions

1. Clone the repository:
   git clone https://github.com/bdtinh1999/Capstone-Project.git
  
2. Install the required libraries:
    pip install -r requirements.txt

## Blog Post :  https://medium.com/@bdtinh1999/predicting-user-churn-for-sparkify-using-spark-mllib-fe724fd7daaa