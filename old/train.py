import json
import pickle
import pandas as pd
import seaborn as sns; sns.set()
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class KaggleTrain:

    def __init__(self, X_train, y_train, parameters):
        self.X_train = X_train
        self.y_train = y_train
        self.parameters = parameters

    def gridsearch(self):

        for id in range(len(self.parameters)):

            self.parameters[id]["estimator"] = [eval(str(self.parameters[id]["estimator"][0]))]

        pipe = Pipeline(steps=[('estimator', RandomForestRegressor())])
        
        self.results = GridSearchCV(pipe, 
                                    self.parameters, 
                                    verbose=2, 
                                    n_jobs=6, 
                                    scoring = 'neg_mean_squared_error')

        self.results.fit(self.X_train, self.y_train)
        
        return self.results


    def best(self):

        self.best_method = self.results.best_estimator_.named_steps['estimator'].__class__.__name__
    
        if self.best_method == 'GradientBoostingRegressor':
            
            self.best_model = GradientBoostingRegressor(
                                random_state=1903,
                                loss="huber",
                                learning_rate = self.results.best_params_["estimator__learning_rate"],
                                n_estimators = self.results.best_params_["estimator__n_estimators"])
            
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

    
    def validate(self, X_valid, y_valid):

        self.X_valid = X_valid
        self.y_valid = y_valid

        y_pred_train = self.best_model.predict(self.X_train)
        self.y_pred_valid = self.best_model.predict(self.X_valid)

        self.mse_train = mse(self.y_train, y_pred_train)
        self.mse_valid = mse(self.y_valid, self.y_pred_valid)

        return self.mse_train, self.mse_valid


    def residuals(self, df = pd.DataFrame()):

        self.errors = df
        self.errors["real"] = self.y_valid
        self.errors["pred"] = self.y_pred_valid
        self.errors["res"] = self.y_pred_valid - self.y_valid

        return self.errors


    def corrplot(self):

       sns.scatterplot(x="real", y="pred", data=self.errors).get_figure()


    def hetplot(self):

       sns.scatterplot(x="pred", y="res", data=self.errors).get_figure()


    def save(self, folder = "models/"):

        pkl_name = "{}.pkl".format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))

        pickle.dump(self.best_model, open(folder + pkl_name, 'wb'))