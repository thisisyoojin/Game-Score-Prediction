import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from validation import Base


class RidgeCV(Base):
    
    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__("Ridge", Ridge, X_train, X_val, y_train, y_val)
        self.params = {
            "alpha": [0.1, 0.5, 1.0, 10.0, 20.0],
        }

class LassoCV(Base):
    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__("Lasso", Lasso, X_train, X_val, y_train, y_val)
        self.params = {
            "alpha": [0.1, 0.5, 1.0, 10.0, 20.0],
        }


class KnnCV(Base):
    
    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__("KNN", KNeighborsRegressor, X_train, X_val, y_train, y_val)
        self.params = {
            "n_neighbors": range(5, 11),
        }


class SvrCV(Base):
    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__("SVR", SVR, X_train, X_val, y_train, y_val)
        self.params = {
            "gamma": np.logspace(-3, 2, 6),
            "C": np.logspace(-3, 2, 6),
        }


class ExtraTreeCV(Base):

    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__("Extra Tree", ExtraTreesRegressor, X_train, X_val, y_train, y_val)
        self.params = {
            "n_estimators": [10, 50, 100, 500, 1000],
        }


class RandomForestCV(Base):
    def __init__(self, X_train, X_val, y_train, y_val):
        super().__init__("Random Forest", RandomForestRegressor, X_train, X_val, y_train, y_val)
        self.params = {
            "n_estimators": [10, 20, 30],
            "max_depth": [10, 20, 30]
            # max_depth.append(None)
            # # Minimum number of samples required to split a node
            # min_samples_split = [2, 5, 10]
            # # Minimum number of samples required at each leaf node
            # min_samples_leaf = [1, 2, 4]
            # # Method of selecting samples for training each tree
            # bootstrap = [True, False]
        }