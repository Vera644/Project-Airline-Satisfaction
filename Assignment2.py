#Assignment2
from random import random

import pandas as pd
import numpy as np
import time

from numpy.random import seed
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler

#Load data from CSV file and Shorten Column Name
df = pd.read_csv("airline_passenger_satisfaction.csv", index_col=False)
df.rename(columns={"Departure and Arrival Time Convenience": "DATC"},inplace=True)

#remove ID column
df.drop('ID', axis=1, inplace=True)


#replace empty space in column labels
df.columns = [c.replace(' ', '_') for c in df.columns]


columns_for_label_encoding = ["Class", "Customer_Type", "Gender", "Type_of_Travel", "Satisfaction"]
labelencoder = LabelEncoder()
for column in columns_for_label_encoding:
    df[column] = df[column].astype(str)
    df[column] = labelencoder.fit_transform(df[column])

#Arrange column order
dataFrame = pd.DataFrame(df,columns=['Gender','Customer_Type','Type_of_Travel','Class',
                                     'Age','Flight_Distance','Departure_Delay','Arrival_Delay',
                                     'DATC','Ease_of_Online_Booking','Check-in_Service','Online_Boarding',
                                     'Gate_Location','On-board_Service','Seat_Comfort','Leg_Room_Service',
                                     'Cleanliness','Food_and_Drink','In-flight_Service','In-flight_Wifi_Service',
                                     'In-flight_Entertainment','Baggage_Handling','Satisfaction'])

outlierDf= pd.DataFrame(df, columns=['Age','Flight_Distance','Departure_Delay','Arrival_Delay'])
#print(df.head(10))

df = df[df['Arrival_Delay'] <=500]



#Detect Missing Values
total = dataFrame.isnull().sum().sort_values(ascending=False)
missing_value = pd.concat([total], axis=1, keys=['Total Missing Values'])
#print(missing_value.head())
mean =round(dataFrame['Arrival_Delay'].mean())

#Fill-in Missing Values with Mean
dataFrame['Arrival_Delay'] = dataFrame['Arrival_Delay'].fillna(mean)
#print('Arrival_Delay Mean: ',df['Arrival_Delay'].mean())
#Export to CSV File
dataFrame.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')


#Load Preprocessed Data
df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)
QuantitativeDF = pd.DataFrame(df, columns=['Age','Flight_Distance','Arrival_Delay'])
df = df.drop('Unnamed: 0', axis=1)



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
#Split the Dataset: training-Validation-Test
indep_train, indep_val, dep_train, dep_val = train_test_split(indep_train,dep_train,
                                                              test_size=0.25,random_state=8)


#Normalize Features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(indep_train)
X_test = scaler.fit_transform(indep_test)



#run knn model
def run_KNN(model, X_train, y_train, X_test, y_test, verbose=True,):
    t0 = time.time()
    if verbose == False:
        model.fit(X_train, y_train.ravel(), verbose=0)
    else:
        #grid search with k-fold
        k_range = range(1,14)
        param_grid = dict(n_neighbors=k_range)
        grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', return_train_score=False, verbose=2)
        train_sizes, train_scores, test_scores = learning_curve(model,X_train , y_train.ravel(), cv=5,
                                                                scoring='accuracy', n_jobs=-4,verbose=2,
                                                                train_sizes=np.linspace(0.01, 1.0, 50))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        #validation and training learning curve plot
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--',
                 label='Validation Accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model accuracy')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()

        #fitting the model for grid search
        optimal_knn = grid.fit(X_train, y_train.ravel())
        model=optimal_knn


    print("best params", model.best_estimator_)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    time_taken = time.time() - t0

    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))

    print("\nK-Nearest-Neighbor Classification Report :")
    print(classification_report(y_test, y_pred, digits=5))
    plot_confusion_matrix(model, X_test, y_test, cmap='Blues', normalize='all')
    plot_roc_curve(model, X_test, y_test)
    plt.show()

    return model, accuracy, roc_auc, time_taken

def learningCurveKNN(model,x,y):
    train_sizes, train_scores, test_scores = learning_curve(model_kn, x, y.ravel(), cv=5,
                                                            scoring='accuracy', n_jobs=-4, verbose=2,
                                                            train_sizes=np.linspace(0.01, 1.0, 50))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    print(train_scores[:15])
    # validation and training learning curve plot
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--',
             label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

params_kn = {'n_neighbors':13, 'algorithm': 'kd_tree', 'n_jobs':4,'leaf_size':30,'p':2}
model_kn = KNeighborsClassifier(**params_kn)

model_rf = RandomForestClassifier(**params_rf)

#model_kn, accuracy_kn, roc_auc_kn, tt_kn = run_KNN(model_kn, X_train, dep_train, X_test, dep_test)
t0 = time.time()
#grid search with k-fold
k_range = range(1,14)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(model_kn, param_grid, cv=5, scoring='roc_auc', return_train_score=False, verbose=2)

print("starting learning curve training")

learningCurveKNN(model_rf,X_train,dep_train)


#fitting the model for grid search
print("_______starting grid-search for KNN_______")
optimal_knn = grid.fit(X_train, dep_train.ravel())
model=optimal_knn


print("best params", model.best_estimator_)
y_pred = model.predict(X_test)

accuracy = accuracy_score(dep_test, y_pred)
roc_auc = roc_auc_score(dep_test, y_pred)
time_taken = time.time() - t0

print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc))
print("Time taken = {}".format(time_taken))

print("\nK-Nearest-Neighbor Classification Report :")
print(classification_report(dep_test, y_pred, digits=5))
plot_confusion_matrix(model, X_test, dep_test, cmap='Blues', normalize='all')
plot_roc_curve(model, X_test, dep_test)
plt.show()

#run Naive Bayes without grid-search or k-fold
def run_NB(model, X_train, y_train, X_test, y_test, verbose=True):
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


#params_nb = {}

#model_nb = CategoricalNB(**params_nb)
#model_nb, accuracy_nb, roc_auc_nb, tt_nb = run_NB(model_nb, X_train, dep_train.ravel(), X_test,
                                                     #dep_test.ravel())

#Naive Bayes Model
model_nb = GaussianNB()
model_nb.fit(X_train, dep_train.ravel())

Y_preds = model_nb.predict(X_test)


print('\nTest Accuracy : {:.3f}'.format(model_nb.score(X_test, dep_test)))
print('Training Accuracy : {:.3f}'.format(model_nb.score(X_train, dep_train)))

print("Accuracy Score : {:.3f}".format(accuracy_score(dep_test, Y_preds)))
print("\nNaive Bayes Classification Report :")
print(classification_report(dep_test, Y_preds))


params = {}

best_nb = GridSearchCV(estimator=GaussianNB(), param_grid=params, n_jobs=-1, cv=10, verbose=5, error_score="raise")
best_nb.fit(X_train, dep_train.ravel())

Y_preds = best_nb.best_estimator_.predict(X_test)
Y_preds_train = best_nb.best_estimator_.predict(X_train)

print('Best Accuracy Through Grid Search : {:.3f}'.format(best_nb.best_score_))
print('Best Parameters : {}\n'.format(best_nb.best_params_))
print("Test Accuracy Score : {:.3f}".format(accuracy_score(dep_test, Y_preds)))
print("Train Accuracy Score : {:.3f}".format(accuracy_score(dep_train, Y_preds_train)))
print("\nClassification Report :")
print(classification_report(dep_test, Y_preds))


cm = confusion_matrix(dep_test, Y_preds, labels=model_nb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_nb.classes_)
disp.plot()
plt.show()

plot_roc_curve(best_nb, X_test, dep_test)



#Hypothesis testing for statistical significance

print("__________________________Hypothesis Test_________________________________________")
seed(1)
scores1 = cross_val_score(best_nb, X_train, dep_train.ravel(), error_score='raise',scoring='accuracy', cv=10, n_jobs=-1, verbose=2)

scores2 = cross_val_score(model_kn, X_train, dep_train.ravel(), scoring='accuracy', cv=10, n_jobs=-1, verbose=2)

print('KNN cross_val training scores: ', scores1)
print('NB cross_val training scores: ', scores2)

#check if difference between algorithms performance
stat, p = wilcoxon(scores1, scores2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Performance is the same (fail to reject H0)')
else:
	print('There is enough evidence to conclude a difference in algorithm performance(reject H0)')