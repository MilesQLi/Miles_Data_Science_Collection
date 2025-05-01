


def immput_median(data, features):
    """
    Impute missing values in the specified features of a DataFrame with the median of each feature.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the features to be imputed.
        features (list): A list of feature names (strings) to be imputed.
        
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    for feature in features:
        if data[feature].isna().sum() > 0:
            median_value = data[feature].median()
            data[feature].fillna(median_value, inplace=True)
    return data

def impute_mode(data, features):
    """
    Impute missing values in the specified features of a DataFrame with the mode of each feature.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the features to be imputed.
        features (list): A list of feature names (strings) to be imputed.
        
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    for feature in features:
        if data[feature].isna().sum() > 0:
            mode_value = data[feature].mode()[0]
            data[feature].fillna(mode_value, inplace=True)
    return data

def impute_group_median(data, features, group_cols):
    """
    Impute missing values in the specified features of a DataFrame with the median of each feature, grouped by specified columns.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the features to be imputed.
        features (list): A list of feature names (strings) to be imputed.
        group_cols (list): A list of column names (strings) to group by.
        
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    for feature in features:
        if data[feature].isna().sum() > 0:
            data[feature] = data.groupby(group_cols)[feature].transform(lambda x: x.fillna(x.median()))
    return data

def impute_group_mode(data, features, group_cols):
    """
    Impute missing values in the specified features of a DataFrame with the mode of each feature, grouped by specified columns.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the features to be imputed.
        features (list): A list of feature names (strings) to be imputed.
        group_cols (list): A list of column names (strings) to group by.
        
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    for feature in features:
        if data[feature].isna().sum() > 0:
            data[feature] = data.groupby(group_cols)[feature].transform(lambda x: x.fillna(x.mode()[0]))
    return data



# Impute missing values using a machine learning model
#From  https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368
class Impute_With_Model:
    
    def __init__(self, model = 'LassoCV', na_frac=0.5, min_samples=0):
        self.model_dict = {}
        self.mean_dict = {}
        self.features = None
        self.na_frac = na_frac
        self.min_samples = min_samples
        if model == 'LassoCV':
            from sklearn.linear_model import LassoCV
            self.model = LassoCV(cv=5, random_state=42)
        elif model == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model == 'DecisionTree':
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(random_state=42)
        
    def find_features(self, data, feature, tmp_features):
        missing_rows = data[feature].isna()
        na_fraction = data[missing_rows][tmp_features].isna().mean(axis=0)
        valid_features = np.array(tmp_features)[na_fraction <= self.na_frac]
        return valid_features

    def fit_models(self, data, features):
        model = self.model
        self.features = features
        n_data = data.shape[0]
        for feature in features:
            self.mean_dict[feature] = np.mean(data[feature])
        for feature in tqdm(features):
            if data[feature].isna().sum() > 0:
                model_clone = clone(model)
                X = data[data[feature].notna()].copy()
                tmp_features = [f for f in features if f != feature]
                tmp_features = self.find_features(data, feature, tmp_features)
                if len(tmp_features) >= 1 and X.shape[0] > self.min_samples:
                    for f in tmp_features:
                        X[f] = X[f].fillna(self.mean_dict[f])
                    model_clone.fit(X[tmp_features], X[feature])
                    self.model_dict[feature] = (model_clone, tmp_features.copy())
                else:
                    self.model_dict[feature] = ("mean", np.mean(data[feature]))
            
    def impute(self, data):
        imputed_data = data.copy()
        for feature, model in self.model_dict.items():
            missing_rows = imputed_data[feature].isna()
            if missing_rows.any():
                if model[0] == "mean":
                    imputed_data[feature].fillna(model[1], inplace=True)
                else:
                    tmp_features = [f for f in self.features if f != feature]
                    X_missing = data.loc[missing_rows, tmp_features].copy()
                    for f in tmp_features:
                        X_missing[f] = X_missing[f].fillna(self.mean_dict[f])
                    imputed_data.loc[missing_rows, feature] = model[0].predict(X_missing[model[1]])
        return imputed_data
# Example use:
#imputer = Impute_With_Model(na_frac=0.4) 
# na_frac is the maximum fraction of missing values until which a feature is imputed with the model
# if there are more missing values than for example 40% then we revert to mean imputation
#imputer.fit_models(train, features)
#train = imputer.impute(train)
#test = imputer.impute(test)