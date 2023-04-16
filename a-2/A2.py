from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import time
import numpy as np

# Load the data
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=None)

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the three algorithms
algorithms = [RandomForestClassifier(), LogisticRegression(), SVC()]

# Initialize the evaluation measures
training_times = []
accuracies = []
f_measures = []

# Initialize the cross-validation object
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterate through the folds of the cross-validation
for train_index, test_index in skf.split(X, y):
    # print(train_index,test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Iterate through the algorithms
    for algorithm in algorithms:
        # Fit the algorithm on the training data
        start = time.time()
        algorithm.fit(X_train, y_train)
        end = time.time()

        # Record the training time
        training_times.append(end - start)

        # Make predictions on the test data
        y_pred = algorithm.predict(X_test)

        # Record the accuracy and F-measure
        accuracies.append(accuracy_score(y_test, y_pred))
        f_measures.append(f1_score(y_test, y_pred))


fold = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5',
        'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10']
alg = ['Random Forest', 'Logistic Regression', 'SVC']

# Reshape the training times
training_times_array = np.array(training_times)
training_times_table = pd.DataFrame(
    training_times_array.reshape(10, 3), index=fold, columns=alg)

# Reshape the accuracies
accuracies_array = np.array(accuracies)
accuracies_table = pd.DataFrame(
    accuracies_array.reshape(10, 3), index=fold, columns=alg)

# Reshape the F-measures
f_measures_array = np.array(f_measures)
f_measures_table = pd.DataFrame(
    f_measures_array.reshape(10, 3), index=fold, columns=alg)


# Initialize a list of table names
table_names = ['Training Times', 'Accuracies', 'F-Measures']

# Initialize a list of tables
tables = [training_times_table, accuracies_table, f_measures_table]

# Iterate through the tables
for table_name, table in zip(table_names, tables):
    # Print the table name
    print(table_name + ":")
    
    # Compute the average and standard deviation
    table.loc['Average'] = table.mean(axis=0)
    table.loc['Standard Deviation'] = table.std(axis=0)
    
    # Print the table
    print(table)
    print()

training_times_table=training_times_table.drop(['Average','Standard Deviation'])
accuracies_table=accuracies_table.drop(['Average','Standard Deviation'])
f_measures_table=f_measures_table.drop(['Average','Standard Deviation'])

import math
# Function to add rankings to a table
def add_rankings(table,name, strat):
    # Create a copy of the table
    print(f"-----------------------------{name}-----------------------------")
    rankings_table = table.copy()
    k=rankings_table.shape[1]
    n=rankings_table.shape[0]
    # rank_table = pd.DataFrame(index=table.index, columns=table.columns)
    # Iterate through the rows of the table

    for i in range(rankings_table.shape[0]):
        # Get the values for the current row
        row = rankings_table.iloc[i, :]
        
        # Sort the values in ascending order
        sorted_row = row.sort_values(ascending=strat)
        
        # Create a ranking for each value
        ranking = {value: rank for rank, value in enumerate(sorted_row, start=1)}
        
        # print(ranking)
        #Update the values in the table with the rankings
        for j in range(rankings_table.shape[1]):
            value = rankings_table.iloc[i, j]
            rankings_table.iloc[i, j] = f'{value:.6f} ({ranking[value]})'
            # rank_table.iloc[i, j] = ranking[value]

    average_rankings = rankings_table.apply(lambda row: row.str.extract(r'\((\d+)\)').astype(int).mean(axis=0))
    # print(average_rankings)
    total_avg= average_rankings.mean(axis=1).iloc[0]

    # print(total_avg)
    squared_differences = (average_rankings - total_avg)**2
    sum_squared_differences = (squared_differences.sum(axis=1).iloc[0])*n
    # print(sum_squared_differences)

    squ_sum=0
    for i in range(1,k+1):
      squ_sum= squ_sum+ ((i-total_avg)**2)*n 
    
    form_2= squ_sum/(n*(k-1))
    fried_stat=sum_squared_differences/form_2
    rankings_table = rankings_table.append(average_rankings, ignore_index=True)
    print(rankings_table)
    print(f"friedman statistic : {fried_stat}")
    critical_value=7.8 #The critical value for k = 3 and n = 10 at the alpha = 0.05 level is 7.8 as per textbook
    if critical_value<fried_stat:
      print(f'The critical value for k = 3 and n = 10 at the alpha = 0.05 level is 7.8 i.e {fried_stat} > 7.8 ' )
      print("The null hypothesis is rejected that is all Algorithms doesnot perform equally")
    else:
      print(f'The critical value for k = 3 and n = 10 at the alpha = 0.05 level is 7.8 i.e {fried_stat} < 7.8 ' )
      print("We Failed to reject null hypothesis that all Algorithms perform equally")

    #Nemenyi test
    Q=2.343 #textbook value(for alpha = 0.05 and k = 3 ,q alpha=2.343 )
    alpha = 0.05
    #Critical Difference = Q * sqrt((k*(k+1)) / (6*n))
    CD=Q*math.sqrt(k*(k+1)/(6*n))
    print("The critical difference is :",CD)
    # print(average_rankings)
    if abs(average_rankings.at[0,'Random Forest']-average_rankings.at[0,'Logistic Regression'])>CD:
      print("Random Forest and Logistic Regression do not perform equally ")

      
    if abs(average_rankings.at[0,'Random Forest']-average_rankings.at[0,'SVC'])>CD:
      print("SVC and Random Forest do not perform equally ")


    if abs(average_rankings.at[0,'Logistic Regression']-average_rankings.at[0,'SVC'])>CD:
      print("Logistic Regression and SVC do not perform equally ")


# Add rankings to the tables
training_times_rankings = add_rankings(training_times_table,'training times',True)
accuracies_rankings = add_rankings(accuracies_table,'Accuracies',False)
f_measures_rankings=add_rankings(f_measures_table,'F-measure',False)