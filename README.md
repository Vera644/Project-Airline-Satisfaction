# Objective
The goal of this assignment is to determine which independent variables have the most influence on the dependent variable, which is in this case the satisfaction of the passengers. 
# Dataset
The dataset used is the [Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) from Kaggle.
It consists of 129880 rows and 25 columns containing both nominal and quantitative variables. The dataset consists of a column or feature named ‘satisfaction’ which describes the overall satisfaction level of the customer. It has two values, ‘neutral or dissatisfied’ and ‘satisfied’. This satisfaction feature is considered as the label feature since it conveys the overall experience of the customer based on the ratings given for other features. The dataset contains a similar amount of each dependent variable category with a count of 73452 Neutral or Unsatisfied and 56428 Satisfied. which makes it balanced.
# Data Cleaning and Visualization
Data cleaning plays a key role in deriving the output of a machine learning model. Usually data cleaning consists of processes like determining outliers and removing or imputing outliers, removing or replacing missing values, removing duplicate values, removing values with less or no importance. Out of all the variables, 'Arrival Delay' showed to have missing values, a total of 393. These are replaced by the mean value. Data Visualisation plays an important role in understanding the data as it gives an overview of the data before the model implementation. Exploratory Data Analysis is done for the dataset.
For the preprocessing, the following steps have been taken:

**1.	Binary Encode Categorical Values and Label Encode Nominal Values.**
Encoding the categorial variables was necessary because all the values need to be in numerical 	form to be processed by the models

**2.	Replace Empty Spaces with ‘_’ in Column Labels:**
Replacing the empty spaces in the column labels made it easier to select.

**3.	Shorten ‘Departure and Arrival Time Convenience’ label:**
The label for Departure and Arrival Time Convenience was shortened to DATC to visualize the 	heatmap more easily.

**4.	Drop Unnecessary Columns:**
The columns dropped were ID, Unnamed: 0. To prevent multicollinearity, Departure Delay was also removed since it had high correlation with Arrival Delay.

**5.	Detect and Fill Missing values:**
Filling in missing data was done in order to maintain representation and reduce bias for these 	samples.

**6.	Removal of Outliers:**
Outliers from the Arrival Delay were removed because airline delays of more than 500 minutes 	are quite unlikely and could be there by error, the results are shown in Figure 4

**7.	Split dataset in training and testing dataset:**
Lastly the dataset is split into 80% training and 20% testing.    

# Data Analysis
To further understand the dataset, descriptive statistics is applied for the numerical features and outliers were detected for the features that are neither nominal or categorical. Distribution analysis is conducted for features 'Age' and 'Flight Distance'
# Models
Models implemented are:

1. K-Nearest-Neighbor (KNN)
2. Categorical Naive Bayes (CNB)
3. Random Forest (RF)
# Conclusions
For the KNN model the number of neighbors has been chosen to be 5. The model performed well with 67% of the time accurate predictions. For the Naïve Bayes model, the categorical model has been used because the majority of the variables were categorical. This model also performed well with an accuracy of 90%. Random Forest performed with an accuracy of 96%
