import pandas as pd


def feature_eng(data, dataprep_list):
    
    X_columns = []
    
    # Missing to classes
    data[dataprep_list["missing_class"]] = data[dataprep_list["missing_class"]].fillna('missing')
    
    # Missing number to Inf
    
    data[dataprep_list["missing_number_to_inf"]] = data[dataprep_list["missing_number_to_inf"]].fillna('missing')
    
    for var in dataprep_list["missing_number_to_inf"]:
        
        X_columns = X_columns +  [var + '_miss_num']
    
        data[[var + '_miss_num']] = data[[var]].replace("missing", -100 )
    
    # Binary to class
    binary_dummy = pd.get_dummies(data=data[dataprep_list["binary_dummies"]], drop_first = True)
    
    data[list(binary_dummy.columns)] = binary_dummy
    X_columns =  X_columns + list(binary_dummy.columns)
    
    # Factor to number
    for var in dataprep_list["factor_to_number"].keys():
        
        X_columns = X_columns +  [var + '_grade']
    
        data[[var + '_grade']] = data[[var]].\
            replace(dataprep_list["factor_to_number"][var]["order"], 
                    dataprep_list["factor_to_number"][var]["grade"])
    
    # continuous to binary
    for var in dataprep_list["continuous_to_binary"].keys():
        
        X_columns = X_columns +  [var + '_binary']
    
        data[[var + '_binary']] = (data[[var]] > dataprep_list["continuous_to_binary"][var]["threshold"]).astype(int)
    
    # Alteração de escala
    for var in dataprep_list["scale_adjust"].keys():
        
        X_columns = X_columns +  [var + '_adj']
    
        data[[var + '_adj']] = data[[var]] + dataprep_list["scale_adjust"][var]["value"]
    
    # Selected Variables
    X_columns = X_columns + list(binary_dummy.columns) + dataprep_list["identity"]

    
    data = data[X_columns]

    return data