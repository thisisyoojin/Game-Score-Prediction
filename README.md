# Game Score Prediction

Game Score Prediction project is a pipeline for collecting data, creating database, and predicting score from steam game data. The data is collected from steam API and custom made web crawler. Collected data is saved locally first, then preprocessed and saved in cloud(AWS) by a configuration file provided. Preprocessed data is used to create models to predict metacritic score.

Created models will be compared with validation data, and a model with the best performance will be saved and used to predict a score. A detailed process can be found in the [presentation file](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true).

## Business Value

The gaming market is huge.
- There are more than 2.7 billion gamers worldwide.
- The global gaming industry will grow at a CAGR of 12% between 2020-2025.
- The PC gaming market could hit $45.5 billion in 2021.

...but the market is very competitive!
- In January 2019, there were 30,000 games on Steam
- Every day 25 new games are released on Steam

A game development process may take from one month to a few years. It will sum the costs of a development team with rights, devices and software costs, and get the right amount. 

Game developers would love to know what features are attrative users in game design process before investing actual budget and effort.
Publishers have many possible games to publish, they want to know which one will be the most popular one will be.

To solve this issue, we can reate a model to predict a metacritic score based on the features of a game. With analysing the result of prediction, we can get a better view on which features are affecting a game score!


## Workflow
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
- Data collection with Web Crawling and API
- Data cleaning & store locally or in cloud(AWS)
- Prediction with ensenble tree-models/Deep neural networks
- Get best model by comparing multiple models


## Result

With the best model is random forest, show the best result of Test score: 0.7598 (mean accuracy).
the top-10 features are listed below:

Priority | Features | Importance
--- | --- | ---
1 | Developer_mean_enc | 0.545 
2 | Publisher_mean_enc | 0.107
3 | Rating | 0.059
4 | Year | 0.044
5 | Day | 0.035
6 | Day | 0.035
7 | Day | 0.035
8 | Day | 0.035
9 | Day | 0.035
10 | Day | 0.035

It is suggested to use.


## Engineered data

Features
- Genre(hot-encoding): Action, Adventure, Casual, Puzzle, RPG, Simulation, Strategy, Racing, Arcade, Sports
- Theme(hot-encoding): Sci-fi-Mechs, Post-apocalyptic, Retro, Zombies, Military, Fantasy, Historical
- Mood(hot-encoding): Violent, Funny, Horror, Sexual,
- Graphic(hot-encoding): 2D, 3D, Cartoon, Pixel, Realistic, Top-Down, Isometric, First-person, Third-person, Resolution
- Contents(hot-encoding): Story-rich, Open world, Choices Matter, Multiple Endings
- Mechanism(hot-encoding): Fight, Shoot, Combat, Platformer, Hack-and-Slash, Survive, Build-and-Create,
- Players(hot-encoding): Single, Multi_local, Multi_online, Competitve, Co-op
- Price(int):(Currency-GBP)
- Release_Date(datetime)
- Required_Age(int)
- Supported languages(hot-encoding)
- Publishers(list)
- Developers(list)
- PC_minimum_processor(int)
- Achievements_counts(int)
- Package_counts(int)
- DLC_counts(int)
- Early access(binary)
- Indie(binary)

Label
- Metacritic score(int: 0-100)

### Models

Machine Learning models
- Baseline
- KNN
- SVR
- Extra tree
- Random Forest
- XGBoost



## Score
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


## Usage

### Validation

Predict metacritic score
If you want to predict predict metacritic score.

```bash
python main.py score -d demo_data.yaml
```
demo_data then you can see the score.

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
