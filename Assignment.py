#Data Preprocessing

import pandas as pd
import seaborn as sns
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



#Load data from CSV file and Shorten Column Name
df = pd.read_csv("airline_passenger_satisfaction.csv", index_col=False)
df.rename(columns={"Departure and Arrival Time Convenience": "DATC"},inplace=True)
print(df['DATC'].head(10))
print('dataset shape')
print(df.shape)

#remove ID column
df.drop('ID', axis=1, inplace=True)


#replace empty space in column labels
df.columns = [c.replace(' ', '_') for c in df.columns]

#check which variables are categorial and numerical
print('_______________Categorial_Variables_______________')
print(df.select_dtypes(include=['object']).columns)
print(' ')
print('_______________Numerical_Variables________________')
print(df.select_dtypes(include=['int64']).columns)


#Binary Encode Categorial Variables
df.replace(["Neutral or Dissatisfied","Satisfied"], [0, 1], inplace=True)
df.replace(["First-time", "Returning"], [0, 1], inplace=True)
df.replace(["Business", "Personal"], [0, 1], inplace=True)
df.replace(["Male", "Female"], [0, 1], inplace=True)



#Label Encode Class category
labelencoder = LabelEncoder()
df['Class'] = df['Class'].astype(str)
df['Class_'] = labelencoder.fit_transform(df['Class'])
df = df.drop('Class',axis=1)

#Arrange column order
dataFrame = pd.DataFrame(df,columns=['Gender','Customer_Type','Type_of_Travel','Class_',
                                     'Age','Flight_Distance','Departure_Delay','Arrival_Delay',
                                     'DATC','Ease_of_Online_Booking','Check-in_Service','Online_Boarding',
                                     'Gate_Location','On-board_Service','Seat_Comfort','Leg_Room_Service',
                                     'Cleanliness','Food_and_Drink','In-flight_Service','In-flight_Wifi_Service',
                                     'In-flight_Entertainment','Baggage_Handling','Satisfaction'])

outlierDf= pd.DataFrame(df, columns=['Age','Flight_Distance','Departure_Delay','Arrival_Delay'])
print(df.head(10))



#Detect outliers
print('------------------Outlier Table-----------------')
Q1 = outlierDf.quantile(0.25)
Q3 = outlierDf.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print('------------------------------------------------')
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + (1.5 * IQR)


#Detect Missing Values
total = dataFrame.isnull().sum().sort_values(ascending=False)
missing_value = pd.concat([total], axis=1, keys=['Total Missing Values'])
print(missing_value.head())
mean =round(dataFrame['Arrival_Delay'].mean())
print(mean)
#Fill-in Missing Values with Mean
dataFrame['Arrival_Delay'] = dataFrame['Arrival_Delay'].fillna(mean)
print('Arrival_Delay Mean: ',df['Arrival_Delay'].mean())
#Export to CSV File
dataFrame.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')


#Load Preprocessed Data
df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)
QuantitativeDF = pd.DataFrame(df, columns=['Age','Flight_Distance','Arrival_Delay'])
df = df.drop('Unnamed: 0', axis=1)


Q1 = df['Arrival_Delay'].quantile(0.25)
Q3 = df['Arrival_Delay'].quantile(0.75)
IQR = Q3 - Q1
res = Q3 + (1.5 * IQR)
print(res)
#Display Removal of outliers
fig, axes = plt.subplots(1, 2)
sns.scatterplot(x="Arrival_Delay", y = np.random.rand(df.shape[0]), data=df,hue="Arrival_Delay",ax=axes[0])


df = df[df['Arrival_Delay'] <=500]

sns.scatterplot(x="Arrival_Delay", y = np.random.rand(df.shape[0]), data=df, hue="Arrival_Delay",ax=axes[1],)
sns.set(rc={'figure.figsize':(30,40)})
plt.show()

#Display Flight_Distance and Age Distribution
figure, axes = plt.subplots(1, 2)
sns.histplot(df,x='Flight_Distance',ax=axes[0])
sns.histplot(df,x='Age',ax=axes[1])
plt.show()

#Descriptive Statistics of Quantitative Variables
print('Descriptive Statistics')
print(QuantitativeDF.describe())
print(QuantitativeDF.describe().to_csv('Descriptive Statistics.csv'))

#Check if dataset is balanced
count = df['Satisfaction'].value_counts()
print(count)
ax = df['Satisfaction'].value_counts(normalize=False).plot(kind='bar',
                                                          color=['indianred','steelblue'],
                                                          alpha=0.9, rot=0,
                                                          figsize=(7, 5.8),
                                                          title="Neutral or Unsatisfied VS Satisfied")
ax.set_xlabel("Satisfied = 1 and Neutral or Unsatisfied = 0")
plt.show()


#Display Correlations
def correlation_matrice(corr,dataFrame):
    corr = dataFrame.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool8))
    f, ax = plt.subplots(figsize=(20, 9))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    annot_kws = {'fontsize': 6}
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None,square=True,
                annot=True,annot_kws=annot_kws ,linewidths=.5)
    plt.show()


#Heatmap Correlation
corr3 = df.corr()
correlation_matrice(corr3,df)

#Update Preprocessed CSV File
df = df.drop('Departure_Delay', axis=1)
df.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')


df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)


#Specify Dependent Variable Columns
independent_var = df.iloc[0:,0:22].values
dependent_var = df.iloc[0:,22:].values
indep_train, indep_test, dep_train, dep_test = train_test_split(independent_var,dependent_var,
                                                                test_size=0.2, random_state=8,
                                                                shuffle=True)
print(indep_train)


def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0 = time.time()
    if verbose == False:
        model.fit(X_train, y_train.ravel(), verbose=0)
    else:
        model.fit(X_train, y_train.ravel())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    time_taken = time.time() - t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))

    print(classification_report(y_test, y_pred, digits=5))
    plot_confusion_matrix(model, X_test, y_test, cmap='Blues', normalize='all')
    plot_roc_curve(model, X_test, y_test)
    plt.show()
    return model, accuracy, roc_auc, time_taken


params_kn = {'n_neighbors':5, 'algorithm': 'kd_tree', 'n_jobs':4}
model_kn = KNeighborsClassifier(**params_kn)
model_kn, accuracy_kn, roc_auc_kn, tt_kn = run_model(model_kn, indep_train, dep_train, indep_test, dep_test)


params_nb = {}

model_nb = CategoricalNB(**params_nb)
model_nb, accuracy_nb, roc_auc_nb, tt_nb = run_model(model_nb, indep_train, dep_train, indep_test, dep_test)
