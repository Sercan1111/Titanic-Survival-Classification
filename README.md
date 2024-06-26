# Titanic Survival Classification Ensemble Learning

## Overview
This project aims to predict the survival of passengers on the Titanic using a dataset of passenger information. The project involves data preprocessing, feature engineering, exploratory data analysis, and classification model training using various machine learning algorithms. The solution is implemented in Python, leveraging libraries such as pandas, scikit-learn, seaborn, and matplotlib.

## Dataset
- train_titanic.csv: The training dataset used for model training and validation.
- test_titanic.csv: The test dataset used for making predictions.
## Preprocessing Steps
1. Data Loading and EDA :
2. Data Cleaning:
3. Correlation Analysis:
4. Feature Scaling:
5. Feature Engineering:


## Visualization
- Bar Plots: Visualize the relationship between independent and dependent features.
- Histograms: Show the distribution of continuous variables.
- Boxplots: Display the spread and outliers in continuous variables.
- Correlation Matrix: Visualize the correlation between different features using a heatmap.
## Classification Models
--Random Forest Classifier
Model Training:
-Train a Random Forest Classifier on the training data.
Model Evaluation:
-Evaluate model performance using accuracy score, classification report, and confusion matrix.
-Visualize feature importances.
## Decision Tree Classifier
-Model Training:
Train a Decision Tree Classifier on the training data.
-Model Evaluation:
-Calculate and visualize the entropy and information gain at the root node.
-Visualize the decision tree.
-Hyperparameter Tuning
Grid Search:
Perform Grid Search to find the best hyperparameters for the Random Forest Classifier.
Model Evaluation:
Evaluate the best model from Grid Search using accuracy score, classification report, and confusion matrix.
Ensemble Model
Voting Classifier:
Combine multiple classifiers (Random Forest, Gradient Boosting, Logistic Regression) using a Voting Classifier.
Model Evaluation:
Evaluate the ensemble model using classification report and confusion matrix.

## Files in the Repository
- titanic_survival_classification.py: The main Python script with all the preprocessing, EDA, modeling, and evaluation steps.
- requirements.txt: Lists all the dependencies to run the script.
- train_titanic.csv: The training dataset.
- test_titanic.csv: The test dataset.
## Results
--Random Forest Classifier
Achieved significant accuracy in predicting passenger survival.
Identified important features contributing to the predictions.
--Decision Tree Classifier
Visualized the decision-making process of the tree.
Calculated information gain at the root node.
--Hyperparameter Tuning
Found the best hyperparameters using Grid Search.
Improved model performance with optimal parameters.
--Ensemble Model
Combined the strengths of multiple classifiers for better performance.
Evaluated the ensemble model with comprehensive metrics.
## Future Work
- Explore more advanced feature engineering techniques.
- Implement additional models and ensemble methods for improved accuracy.
- Integrate the models into a real-time prediction system for practical use.
## Contributions
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
