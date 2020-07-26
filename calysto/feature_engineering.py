import numpy as np
import pandas as pd

class Engineering:

    def __init__(self, data):
        self.data = data


    def target(self, target_id):
        self.target_id = target_id
        target = np.log(np.ravel(self.data[[self.target_id]]))   
        return target


    def detarget(self, y_pred_test):
        self.y_pred_test = y_pred_test
        target = np.exp(self.y_pred_test)
        return target


    def feature(self, params):
        self.params = params

        df = self.data
        
        X_columns = []
        
        # Missing to classes
        df[self.params["missing_class"]] = df[self.params["missing_class"]].fillna('missing')
        
        # Missing number to Inf
        
        df[self.params["missing_number_to_inf"]] = df[self.params["missing_number_to_inf"]].fillna('missing')
        
        for var in self.params["missing_number_to_inf"]:
            
            X_columns = X_columns +  [var + '_miss_num']
        
            df[[var + '_miss_num']] = df[[var]].replace("missing", -100 )
        
        # Binary to class
        binary_dummy = pd.get_dummies(data=df[self.params["binary_dummies"]], drop_first = True)
        
        df[list(binary_dummy.columns)] = binary_dummy
        X_columns =  X_columns + list(binary_dummy.columns)

        #Dummy controlada
        for var in self.params["dummy_controlada"].keys():
            for classe in list(self.params["dummy_controlada"][var].keys()):
                new_var = str(var) + "_" + str(classe)
                X_columns = X_columns + [new_var]
                df[new_var] = df[var].fillna("missing").isin(self.params["dummy_controlada"][var][classe])

        #Dummy faltante
        for var in self.params["insert_dummy_faltante"].keys():
            
            dummy_faltante = pd.get_dummies(data=df[var], drop_first = True)

            if self.params["insert_dummy_faltante"][var]["dummy"] in dummy_faltante.columns:
                a = 1
            else:
                dummy_faltante[self.params["insert_dummy_faltante"][var]["dummy"]] = 0

            df[list(dummy_faltante.columns)] = dummy_faltante
            X_columns =  X_columns + list(dummy_faltante.columns)
        
        # Factor to number
        df[list(self.params["factor_to_number"].keys())] = df[list(self.params["factor_to_number"].keys())].fillna('missing')
        for var in self.params["factor_to_number"].keys():
            
            X_columns = X_columns +  [var + '_grade']
        
            df[[var + '_grade']] = df[[var]].\
                replace(self.params["factor_to_number"][var]["order"], 
                        self.params["factor_to_number"][var]["grade"])
        
        # continuous to binary
        for var in self.params["continuous_to_binary"].keys():
            
            X_columns = X_columns +  [var + '_binary']
        
            df[[var + '_binary']] = (df[[var]] > self.params["continuous_to_binary"][var]["threshold"]).astype(int)
        
        # Alteração de escala
        for var in self.params["scale_adjust"].keys():
            
            X_columns = X_columns +  [var + '_adj']
        
            df[[var + '_adj']] = df[[var]] + self.params["scale_adjust"][var]["value"]

        # Unify to class
        for var in self.params["unify_classes"].keys():
            
            X_columns = X_columns +  [var + '_unify']

            class1 = self.params["unify_classes"][var]["class"]
            df.loc[df[var] != class1, var] = 0
            df.loc[df[var] == class1, var] = 1

            df[[var + '_unify']] = df[[var]]


        # Selected Variables
        X_columns = X_columns + list(binary_dummy.columns) + self.params["identity"]

        
        feature_table = df[set(X_columns)]

        return feature_table