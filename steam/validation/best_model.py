import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def set_baseline(X_train, X_val, y_train, y_val):

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    val_loss = np.mean((y_val - y_pred)**2)
    print("Baseline loss:", val_loss)
    print("Baseline score:", lr.score(X_val, y_val))

    return lr.score(X_val, y_val)



def get_best_models(models, X_train, X_val, y_train, y_val):

    results = []

    for model in models:
        reg = model(X_train, X_val, y_train, y_val)
        best_model, info = reg.best_model()
        best = {
            "model_name": info['model_name'],
            "best_model": best_model,
            "train_loss": info['train_loss'],
            "val_loss": info['val_loss'],
            "train_score": info['train_score'],
            "val_score": info['val_score']
        }
        results.append(best)
    
    return results





def plot_train_val(results, type='loss', label='Loss(MSE)'):

    N = len(results)
    train_loss = [r[f'train_{type}'] for r in results]
    val_loss = [r[f'val_{type}'] for r in results]
    names = [r['model_name'] for r in results]

    index = np.arange(N)
    width = 0.35
    plt.bar(index, train_loss, width, label=f'train_{type}')
    plt.bar(index + width, val_loss, width, label=f'val_{type}')

    plt.xticks(index + width / 2, names)
    plt.ylabel(label)
    plt.xlabel('Models')
    plt.legend(loc='lower right')
    plt.show()

