import json
import pickle
import pandas as pd
import seaborn as sns; sns.set()
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy.stats import uniform, randint


class Training:

    def __init__(self, X, y, parameters):
        self.X = X
        self.y = y
        self.parameters = parameters

    def train_test_split(self):

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, test_size=0.2, random_state=1903)

    def gridsearch(self, random=False):
        self.random = random

        for id in range(len(self.parameters)):

            self.parameters[id]["estimator"] = [eval(str(self.parameters[id]["estimator"][0]))]

        pipe = Pipeline(steps=[('estimator', GradientBoostingRegressor())])
        
        if self.random:

            distributions = dict(estimator__n_estimators=randint(low=500, high=1000),
                                         estimator__learning_rate=uniform(loc=0.01, scale=1),
                                         estimator__subsample = uniform(loc=0.01, scale=0.9),
                                         estimator__min_samples_split = randint(low=1, high=5),
                                         estimator__min_samples_leaf = randint(low=1, high=10),
                                         estimator__min_impurity_decrease = uniform(loc=0, scale=0.9),
                                         estimator__alpha = uniform(loc=0.01, scale=0.5),
                                         estimator__max_depth = randint(low=1, high=10)
                                         )

            self.results = RandomizedSearchCV(pipe,
                                    distributions,
                                    n_iter = 150, 
                                    verbose=3, 
                                    n_jobs=-1, 
                                    scoring = 'neg_mean_squared_error')

        else: 
            
            self.results = GridSearchCV(pipe, 
                                    self.parameters, 
                                    verbose=3, 
                                    n_jobs=-1,
                                    scoring = 'neg_mean_squared_error')

        self.results.fit(self.X_train, self.y_train)
        
        return self.results


    def best(self):

        self.best_method = self.results.best_estimator_.named_steps['estimator'].__class__.__name__
    
        if self.best_method == 'GradientBoostingRegressor':
            
            self.best_model = GradientBoostingRegressor(
                                random_state=self.results.best_params_["estimator__random_state"],
                                loss="huber",
                                learning_rate = self.results.best_params_["estimator__learning_rate"],
                                n_estimators = self.results.best_params_["estimator__n_estimators"],
                                subsample = self.results.best_params_["estimator__subsample"],
                                min_samples_split = self.results.best_params_["estimator__min_samples_split"],
                                min_samples_leaf = self.results.best_params_["estimator__min_samples_leaf"],
                                min_impurity_decrease = self.results.best_params_["estimator__min_impurity_decrease"],
                                alpha = self.results.best_params_["estimator__alpha"],
                                max_depth = self.results.best_params_["estimator__max_depth"]
                                )
                
        else:
            if self.best_method == 'RandomForestRegressor':

                self.best_model = RandomForestRegressor(
                                    criterion = "mse", 
                                    random_state=0,
                                    max_features = self.results.best_params_["estimator__max_features"],
                                    n_estimators = self.results.best_params_["estimator__n_estimators"],
                                    min_samples_split = self.results.best_params_["estimator__min_samples_split"],
                                    ccp_alpha = self.results.best_params_["estimator__ccp_alpha"],
                                    min_samples_leaf = self.results.best_params_["estimator__min_samples_leaf"])

            else:

                self.best_model = None
        
        self.best_model.fit(self.X_train, self.y_train)
        
        return self.best_model

    
    def metrics(self):

        y_pred_train = self.best_model.predict(self.X_train)

        self.mse_train = mse(self.y_train, y_pred_train)

        return self.mse_train

    def save(self, folder = "models/"):

        pkl_name = "{}.pkl".format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))

        pickle.dump(self.best_model, open(folder + pkl_name, 'wb'))


    def validate(self):

        self.y_pred_valid = self.best_model.predict(self.X_valid)

        self.mse_valid = mse(self.y_valid, self.y_pred_valid)

        return self.mse_valid


    def residuals(self, df = pd.DataFrame()):

        self.errors = df
        self.errors["real"] = self.y_valid
        self.errors["pred"] = self.y_pred_valid
        self.errors["res"] = self.y_pred_valid - self.y_valid

        return self.errors


    def corrplot(self):

       sns.scatterplot(x="real", y="pred", data=self.residuals())


    def hetplot(self):

       sns.scatterplot(x="pred", y="res", data=self.residuals())