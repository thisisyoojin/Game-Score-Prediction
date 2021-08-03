#%%
import numpy as np
import pandas as pd

class Preprocessor:
    
    def __init__(self):
        self.norm_params = {}
        self.one_hot_params = {}
        self.fe_params = {}
        self.me_params = {}
        self.features = []
        

    def normalise_data(self, X, col):

        params = self.norm_params.get(col)

        if params is None:
            self.norm_params[col] = {
                "mean": np.mean(X[col]),
                "std": np.std(X[col])
            }

        X.loc[:, col] = X[col].apply(lambda x: (x - self.norm_params[col]['mean']) / self.norm_params[col]['std']) 
        


    def one_hot_encoding(self, X, col):
        
        params = self.one_hot_params.get(col)
        
        if params is None:
            self.one_hot_params[col] = X[col].unique()
        
        for uq in self.one_hot_params[col]:
            X.loc[:, f"{col}_{uq}"] = X[col].apply(lambda x: 1 if x == uq else 0)
            self.features.append(f"{col}_{uq}")
        
        return X


    def frequency_encoding(self, X, col):
        
        params = self.fe_params.get(col)
        
        if params is None:
            self.fe_params[col] = X.groupby(col).size() / len(X)
        
        X.loc[:, f"{col}_freq_enc"] = X[col].map(self.fe_params[col])
        X[f"{col}_freq_enc"].fillna(0, inplace=True)
        self.normalise_data(X, f'{col}_freq_enc')
        self.features.append(f'{col}_freq_enc')

        return X


    def mean_encoding(self, X, y, col):
        
        params = self.me_params.get(col)

        if params is None:
            df = pd.concat([X, y], axis=1)
            self.me_params[col] = df.groupby(col)['Target'].mean()
        
        X.loc[:, f"{col}_mean_enc"] = X[col].map(self.me_params[col])
        X[f'{col}_mean_enc'].fillna(0, inplace=True)
        self.normalise_data(X, f'{col}_mean_enc')
        self.features.append(f'{col}_mean_enc')
        
        return X


    def preprocess_required_age(self, X):
        
        def decide_group(row):
            if row < 8:
                return 0
            elif row < 17:
                return 1
            else:
                return 2
        
        X.loc[:, 'Age_group'] = X['Required_Age'].apply(decide_group)
        self.normalise_data(X, 'Rating')
        self.features.extend(['Required_Age', 'Age_group'])
        
        return X

    def preprocess_rating(self, X):
        
        def decide_rating(row):
            if row < 0:
                return 0
            elif row < 50:
                return 1
            else:
                return 2

        X.loc[:, 'Rating_group'] = X['Rating'].apply(decide_rating)

        self.features.extend(['Rating', 'Rating_group'])

        return X



    def preprocess_release_date(self, X):

        X['Year'] = pd.DatetimeIndex(X['Release_Date']).year
        X['Year'] = X['Year'] - X['Year'].min()

        X['Month'] = pd.DatetimeIndex(X['Release_Date']).month
        X['Day'] = pd.DatetimeIndex(X['Release_Date']).day

        X['Weekend'] = pd.DatetimeIndex(X['Release_Date']).weekday > 4
        X.loc[:, 'Weekend'] = X['Weekend'].apply(lambda x: 1 if True else 0)

        self.normalise_data(X, 'Year')
        self.normalise_data(X, 'Month')
        self.normalise_data(X, 'Day')
        
        self.features.extend(['Year', 'Month', 'Day', 'Weekend'])
        
        return X




    def preprocess(self, X, y):

        # numerical values
        X["Achievement_rate"].fillna(0, inplace=True)
        self.normalise_data(X, 'Achievement_rate')
        self.features.append("Achievement_rate")

        self.features.append("Is_Multiplayer")
        
        self.normalise_data(X, 'Price')
        self.features.append('Price')

        self.preprocess_rating(X)
        
        # categorical values
        X = self.preprocess_required_age(X)
        X = self.one_hot_encoding(X, 'Type')
        
        X["Genre"].fillna("Unknown", inplace=True)
        X["Developer"].fillna("Unknown", inplace=True)

        X = self.frequency_encoding(X, 'Genre')
        X = self.frequency_encoding(X, 'Developer')
        X = self.frequency_encoding(X, 'Publisher')
  
        X = self.mean_encoding(X, y, 'Genre')
        X = self.mean_encoding(X, y, 'Developer')
        X = self.mean_encoding(X, y, 'Publisher')

        # date values
        X = self.preprocess_release_date(X)

        return X[self.features]


    def fit(self, X_train, y_train):
        """
        Training dataset
        """
        return self.preprocess(X_train, y_train)
        

    def transform(self, X_test):
        """
        validation set, test set
        """
        self.features = []
        return self.preprocess(X_test, y=None)


