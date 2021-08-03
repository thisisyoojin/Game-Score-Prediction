from sklearn.model_selection import train_test_split
from database import create_df
from preprocess import Preprocessor


def prepare_dataset(kind='cv'):

    X, y = create_df()
    pc = Preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

    if kind == 'cv':
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=33)
        X_train = pc.fit(X_train, y_train)
        X_val = pc.transform(X_val)
        return X_train, X_val, y_train, y_val

    else:
        X_train = pc.fit(X_train, y_train)
        X_test = pc.transform(X_test)
        return X_train, X_test, y_train, y_test

