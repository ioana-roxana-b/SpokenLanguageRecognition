import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
import model
def minmax_sc(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def stand_sc(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def classification(c):

    # data = pd.read_csv("more_features.csv")
    data = dd.read_csv("more_features.csv", sample=2**30)

    # Convert Dask DataFrame to Pandas DataFrame
    data_pandas = data.compute()

    # Split the data into features and labels
    X = data_pandas.drop('label', axis=1)
    y = data_pandas['label']

    # Split the data using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    print("AAAAAA")
    #X_train, X_test = minmax_sc(X_train, X_test)
    X_train, X_test = stand_sc(X_train, X_test)

    clf, y_pred, clf_name = model.pick(X_train, y_train, X_test, c)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    results_df = pd.DataFrame({
        'Classifier': [clf_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })

    results_df.to_csv(f'results_{clf_name}.csv', mode='a', index=False)
