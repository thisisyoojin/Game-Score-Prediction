# Predict steam sales & playtime

## Project brief

The gaming market is huge
- There are more than 2.7 billion gamers worldwide.
- The global gaming industry will grow at a CAGR of 12% between 2020-2025.
- The PC gaming market could hit $45.5 billion in 2021.

...but very competitive
- In January 2019, there were 30,000 games on Steam
- Every day 25 new games are released on Steam


Game developers have skills to develop a game, but they are not sure what game players love to play.
Publishers have many possible games to publish, but they want to know which one will sell the most.

To solve this issue,
we can reate a model to predict the average sales and playtime based on the features of a game!


## Data

I used steam dataset for this project, which can be found here > https://steam.internet.byu.edu/
Data from multiple tables in the dataset is used to create the dataframe for this proejct.

The dataframe has following features:

- Type
- Price
- Rating
- Release_Date
- Required_Age
- Is_Multiplayer
- Genre
- Achievement Percentage
- Publisher
- Developer

With these features, we are going to predict the average of sales and playtime of the game.
-Target: (Total_buyers + Playtime)/2

## Results

### Validation
Models | Val_score
--- | --- 
Baseline | 0.412461270
Ridge | 0.412298260
Lasso | 0.423950403
KNN | 0.239745888
SVR | 0.440926652
Extra Tree | 0.435693013
Random Forest | 0.440926652

Random Forest model shows the best result

### Best model

- Random Forest

- Test score: 0.7598 (mean accuracy)

- Feature importance

Features | Importance
--- | --- 
Developer_mean_enc | 0.545 
Publisher_mean_enc | 0.107
Rating | 0.059
Year | 0.044
Day | 0.035

## Usage

### Validation
```
from preprocess import prepare_dataset
from validation.models import RidgeCV, LassoCV, KnnCV, SvrCV, RandomForestCV, ExtraTreeCV

X_train, X_val, y_train, y_val = prepare_dataset(kind='cv')

baseline = set_baseline(X_train, X_val, y_train, y_val)

models = [RidgeCV, LassoCV, KnnCV, SvrCV, RandomForestCV, ExtraTreeCV]
results = get_best_models(models, X_train, X_val, y_train, y_val)
winner_model = sorted(results, key=lambda x: x['val_score'], reverse=True)[0]['best_model']
```

### Prediction
```
from validation import set_baseline, get_best_models, plot_train_val

X_train, X_val, y_train, y_val = prepare_dataset(kind='train')
winner_model.fit(X_train, y_train)
print(winner_model.score(X_test, y_test))
```
