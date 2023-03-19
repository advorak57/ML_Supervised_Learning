# Module 12 Report Template

## Overview of the Analysis


The purpose of this analysis was to build and evaluate machine learning models for credit risk classification using a dataset of historical lending activity from a peer-to-peer lending services company. The dataset was imbalanced, with healthy loans outnumbering high-risk loans. The objective was to predict whether a loan was healthy (0) or high-risk (1) based on the other available features in the dataset.

### The stages of the machine learning process were as follows:

### Data preparation: 

The lending_data.csv data from the Resources folder was read into a Pandas DataFrame, and the labels set (y) was created from the “loan_status” column, while the features (X) DataFrame was created from the remaining columns. The balance of the labels variable (y) was checked using the value_counts function, and the data was split into training and testing datasets by using train_test_split.

### Model building and evaluation: Two models were built and evaluated:

A logistic regression model was built with the original data, and its performance was evaluated using accuracy score, confusion matrix, and classification report.
A logistic regression model was built with resampled data, using the RandomOverSampler module from the imbalanced-learn library to ensure that the labels had an equal number of data points. Its performance was also evaluated using accuracy score, confusion matrix, and classification report.
Results
The results of the machine learning models are as follows:

## Machine Learning Model 1 (Logistic Regression Model with Original Data):

Accuracy: 0.99
Precision and Recall for Label 0 (Healthy Loan): 1.00
Precision and Recall for Label 1 (High-Risk Loan): 0.87 and 0.89, respectively
F1-score for Label 0: 1.00
F1-score for Label 1: 0.88


## Machine Learning Model 2 (Logistic Regression Model with Resampled Data):

Accuracy: 1.00
Precision and Recall for Label 0 (Healthy Loan): 1.00
Precision and Recall for Label 1 (High-Risk Loan): 0.87 and 1.00, respectively
F1-score for Label 0: 1.00
F1-score for Label 1: 0.93

## Summary

Both machine learning models performed very well in predicting both the 0 (healthy loan) and 1 (high-risk loan) labels, as evidenced by their high accuracy scores, high precision and recall values for both labels, and high F1-scores. However, the model built with resampled data performed slightly better, with a perfect recall score for Label 1, indicating that it was able to identify all the actual high-risk loans.

In terms of which model to use, the logistic regression model built with resampled data seems to be the better choice since it achieved better recall for Label 1, which is the label we want to predict accurately. This model is recommended for identifying high-risk loans since it's better at catching them. However, the original model could also be used, especially if recall for Label 1 is not a crucial metric for the problem at hand.

It's worth noting that the choice of which model to use ultimately depends on the specific requirements of the business problem being addressed. If correctly identifying high-risk loans is of utmost importance, the resampled model should be preferred, whereas if a balance between high-risk and healthy loans is more critical, the original model may suffice.