import pandas as pd
import numpy as np
from time import perf_counter, time
import itertools
from multiprocessing import Process

class Base:

    def __init__(self, name, model, X_train, X_val, y_train, y_val):
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.name = name
        self.model = model
        self.params = {}
        self.results = None
        

    def generate_params(self, params):
        
        keys = list(params.keys())
        values = params.values()
        grid_params = list(itertools.product(*values))

        return keys, grid_params


    def loss_mse(self, y_true, y_predict):
        return np.mean((y_true - y_predict)**2)


    def tune_model(self):

        self.results = []
        keys, grid_params = self.generate_params(self.params)

        for r in grid_params:
            param = {keys[idx]:r[idx] for idx in range(len(r))}
            result = self.fit(param)
            if result is not None:
                self.results.append(result)
        
        return self.results


    def best_model(self):

        if self.results is None:
            self.results = self.tune_model()

        results_df = pd.DataFrame(self.results)
        best_result = results_df[results_df['val_score'] == results_df['val_score'].max()].iloc[0]
        best_params = best_result.params
        
        return self.model(**best_params), best_result



    def fit(self, param):
        
        reg = self.model(**param)

        TIMEOUT = 60

        start = perf_counter()
        print(f"Start training {self.name} with {param}")
        reg.fit(self.X_train, self.y_train)
        # pr = Process(target=reg.fit, args=(self.X_train, self.y_train))
        # pr.start()
        # pr.join()
        # while perf_counter() - start <= TIMEOUT:
        #     if not pr.is_alive():
        #         print('??')
        #         break
        # pr.terminate()
        end = perf_counter()
        
        # if pr.exitcode is None:
        #     print(f"Failed training {self.name} with {param}")
        #     return

        train_loss = self.loss_mse(self.y_train, reg.predict(self.X_train))
        val_loss = self.loss_mse(self.y_val, reg.predict(self.X_val))
        

        result = {
            'model_name': self.name,
            'params': param,
            'time_taken' : (end-start),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_score': reg.score(self.X_train, self.y_train),
            'val_score': reg.score(self.X_val, self.y_val),
        }

        return result


