
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # For potential early stopping, though not strictly required here

def impute_target_with_mlp(df: pd.DataFrame, target_column: str, feature_columns: list) -> pd.DataFrame:
    """
    Imputes missing values in a target column of a Pandas DataFrame using MLPRegressor.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the column with missing values to be imputed.
        feature_columns (list): A list of column names to be used as features for imputation.

    Returns:
        pd.DataFrame: A new DataFrame with the target column's missing values imputed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(target_column, str):
        raise TypeError("Input 'target_column' must be a string.")
    if not isinstance(feature_columns, list) or not all(isinstance(col, str) for col in feature_columns):
        raise TypeError("Input 'feature_columns' must be a list of strings.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in DataFrame.")
    if target_column in feature_columns:
        print(f"Warning: Target column '{target_column}' is also in feature_columns. "
              "This might lead to data leakage if not handled carefully during training/prediction separation.")

    df_imputed = df.copy()

    # Identify rows with missing and non-missing target values
    missing_mask = df_imputed[target_column].isnull()
    
    # If no missing values, return the original dataframe copy
    if not missing_mask.any():
        print(f"No missing values found in the target column '{target_column}'.")
        return df_imputed

    train_df = df_imputed[~missing_mask]
    predict_df = df_imputed[missing_mask]

    # If no data to train on, cannot impute
    if train_df.empty:
        print(f"No non-missing values in '{target_column}' to train the imputer. Returning original DataFrame.")
        return df_imputed
        
    # If no features provided, cannot use MLP
    if not feature_columns:
        print("No feature columns provided. Cannot use MLP for imputation. Returning original DataFrame.")
        # Optionally, you could fall back to simple mean/median imputation of the target here
        # target_mean = df_imputed[target_column].mean() # or median()
        # df_imputed[target_column].fillna(target_mean, inplace=True)
        return df_imputed

    X_train_raw = train_df[feature_columns]
    y_train = train_df[target_column]
    X_predict_raw = predict_df[feature_columns]
    
    # If there's nothing to predict (e.g. all values were non-missing originally, though caught above)
    if X_predict_raw.empty:
        return df_imputed # Should be caught by `if not missing_mask.any():`

    # 1. Impute missing values in FEATURES (if any)
    # Important: Fit imputer on training features only, then transform both train and predict features
    feature_imputer = SimpleImputer(strategy='mean') 
    X_train_imputed_features = feature_imputer.fit_transform(X_train_raw)
    X_predict_imputed_features = feature_imputer.transform(X_predict_raw)

    # Convert back to DataFrame to keep column names for clarity, though not strictly necessary for scaler
    X_train_imputed_features_df = pd.DataFrame(X_train_imputed_features, columns=feature_columns, index=X_train_raw.index)
    X_predict_imputed_features_df = pd.DataFrame(X_predict_imputed_features, columns=feature_columns, index=X_predict_raw.index)

    # 2. Scale features
    # Important: Fit scaler on training features only, then transform both train and predict features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed_features_df)
    X_predict_scaled = scaler.transform(X_predict_imputed_features_df)

    # 3. Define and train MLP Regressor
    num_features = X_train_scaled.shape[1]
    if num_features == 0: # Should be caught by "if not feature_columns:"
        print("No features available after processing. Cannot train MLP.")
        return df_imputed
        
    hidden_layer_size = max(1, int(num_features / 2)) # Ensure at least 1 unit

    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_size,),
        activation='relu',         # Common choice
        solver='adam',             # Efficient for large datasets
        max_iter=500,              # Increase if model doesn't converge
        random_state=42,           # For reproducibility
        early_stopping=True,       # To prevent overfitting and stop early
        n_iter_no_change=10        # How many iterations with no improvement to wait
    )

    print(f"Training MLPRegressor with hidden_layer_sizes=({hidden_layer_size},) for target '{target_column}'...")
    mlp.fit(X_train_scaled, y_train)

    # 4. Predict missing values
    print("Predicting missing values...")
    predicted_values = mlp.predict(X_predict_scaled)

    # 5. Fill missing values in the copied DataFrame
    df_imputed.loc[missing_mask, target_column] = predicted_values
    print(f"Imputation complete for column '{target_column}'.")

    return df_imputed

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