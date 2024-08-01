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
To further understand the dataset, correlation analysis is conducted

![image](https://github.com/user-attachments/assets/7827ce3a-7d3c-4124-8fb8-e37cb61c141d)

Descriptive statistics is applied for the numerical features and outliers were detected for the features that are neither nominal or categorical.

![descriptive_statistics](https://github.com/user-attachments/assets/43f6720c-5e56-4b70-8aa6-308dbb707cbf)

The distribution of the two variables Flight Distance and Age were further explored. The histograms below show that Flight Distance is heavily skewed right whereas Age does not show significant skewness this could be due to this variable having a low number of outliers. Another observation from the Age distribution is that there is no data from passengers of certain ages. The average flight distance according to the descriptive statistics table is 1190.316 miles, because many countries are much further apart than this value, these outliers will not be removed since it is possible few passengers have traveled to far countries. The outliers for Age will also not be removed since it is a possibility certain flights include elderly passengers.   

![image](https://github.com/user-attachments/assets/8b1161c5-81cf-48d8-8a20-bef560ce06c2)

Below is a scatterplot for Arrival Delay before and after outliers have been removed. The figure shows there are quite a few outliers that indicate flights were delayed up to around 1600 minutes. Because it is quite unlikely that flight delays will be this long and there are few instances that greater than 500 in the chart. Outliers with a value larger than 500 will be removed. 

![image](https://github.com/user-attachments/assets/f10f5a02-546b-4264-adad-ef060fb900dd)


# Models
Models implemented are:

K-Nearest-Neighbor (KNN)
   
![KNN](https://github.com/user-attachments/assets/5c8180d7-95f1-4e9f-8228-af27dbd0229b)

Categorical Naive Bayes (CNB)

![CNB](https://github.com/user-attachments/assets/6db2e25a-ec44-49ea-8c13-b262e22cfd74)

Random Forest (RF)

![image](https://github.com/user-attachments/assets/71bc19d6-56ba-4f15-977e-10fb9497c5ff)

# Conclusions
For the KNN model the number of neighbors has been chosen to be 5. The model performed well with 67% of the time accurate predictions. For the Naïve Bayes model, the categorical model has been used because the majority of the variables were categorical. This model also performed well with an accuracy of 90%. Random Forest performed with an accuracy of 96%
