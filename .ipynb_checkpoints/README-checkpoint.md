# Credit Risk Classification

This project involves the classification of credit risk based on historical lending activity from a peer-to-peer lending services company. The dataset used in this project was inherently imbalanced since healthy loans were significantly more numerous than high-risk loans. The goal of the project was to train and evaluate machine learning models that could predict the creditworthiness of borrowers based on other available features in the dataset.

## Technologies

The project was completed using the following technologies:

* Python 3.8.10
* pandas 1.3.4
* scikit-learn 1.0.2
* imbalanced-learn 0.8.1
* Installation

To run this project, you will need to install the following packages using pip:

pip install pandas

pip install scikit-learn

pip install imbalanced-learn


## Data

The lending_data.csv data was used for this project. The dataset contained 68,817 rows and 27 columns of financial information about borrowers. The labels set (y) was created from the “loan_status” column, while the features (X) DataFrame was created from the remaining columns. A value of 0 in the “loan_status” column meant that the loan was healthy, while a value of 1 meant that the loan had a high risk of defaulting.

## Analysis

The project involved the following steps:

### Data preparation: 

The lending_data.csv data was read into a Pandas DataFrame, and the labels set (y) and features (X) DataFrame were created. The balance of the labels variable (y) was checked, and the data was split into training and testing datasets using train_test_split.

### Model building and evaluation: 

### Two machine learning models were built and evaluated:

A logistic regression model was built with the original data, and its performance was evaluated using accuracy score, confusion matrix, and classification report.
A logistic regression model was built with resampled data, using the RandomOverSampler module from the imbalanced-learn library to ensure that the labels had an equal number of data points. Its performance was also evaluated using accuracy score, confusion matrix, and classification report.

## Results

Both machine learning models performed well in predicting both the 0 (healthy loan) and 1 (high-risk loan) labels. However, the model built with resampled data performed slightly better, with a perfect recall score for Label 1, indicating that it was able to identify all the actual high-risk loans.

## Summary

The logistic regression model built with resampled data seems to be the better choice since it achieved better recall for Label 1, which is the label we want to predict accurately. This model is recommended for identifying high-risk loans since it's better at catching them. However, the original model could also be used, especially if recall for Label 1 is not a crucial metric for the problem at hand.

Ultimately, the choice of which model to use depends on the specific requirements of the business problem being addressed.