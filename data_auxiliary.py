import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def split_data(X, y, split_ratio):
    X_scaled = pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def standardise(X):
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled)


