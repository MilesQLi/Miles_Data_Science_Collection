import numpy as np


# To reduce memory usage of a pandas DataFrame, you can use the following function. 
# From https://www.kaggle.com/code/pahirathannithilan/music-recommendation-system-letsgrowmore
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# Correcting Skewed feature
def transform_skewed_feature(df, columns, method='sqrt'):
    df2 = df.copy()
    for col in columns:
        if method == 'log':
            df2[col] = np.log1p(df2[col])
        elif method == 'sqrt':
            df2[col] = np.sqrt(df2[col])
    return df2



def find_categorical_mismatches(train_df, test_df, categorical_columns=None):
    """
    Find categorical feature values that appear in the test dataframe but not in the training dataframe.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        The training dataframe used to fit the model
    test_df : pandas.DataFrame
        The test dataframe used for prediction
    categorical_columns : list, optional
        List of categorical column names. If None, will try to infer categorical columns
        by checking for object and category dtypes
        
    Returns:
    --------
    dict:
        A dictionary where keys are column names and values are lists of categories
        that appear in test_df but not in train_df
    """
    import pandas as pd
    
    # If categorical columns not specified, try to infer them
    if categorical_columns is None:
        categorical_columns = set(train_df.select_dtypes(include=['object', 'category']).columns.tolist())
        categorical_columns.intersection_update(test_df.select_dtypes(include=['object', 'category']).columns.tolist())
    
    mismatches = {}
    
    for col in categorical_columns:
        if col not in train_df.columns or col not in test_df.columns:
            print(f"Warning: Column '{col}' not found in both dataframes")
            continue
            
        # Get unique values from both dataframes
        train_categories = set(train_df[col].astype(str).unique())
        test_categories = set(test_df[col].astype(str).unique())
        
        # Find values in test that aren't in train
        new_categories = test_categories - train_categories
        
        if new_categories:
            mismatches[col] = list(new_categories)
    
    return mismatches

def fix_categorical_mismatches(train_df, test_df, categorical_columns=None, strategy='replace_with_common'):
    """
    Fix categorical features in test_df to match the categories in train_df.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        The training dataframe used to fit the model
    test_df : pandas.DataFrame
        The test dataframe used for prediction
    categorical_columns : list, optional
        List of categorical column names. If None, will infer categorical columns
    strategy : str, default='replace_with_common'
        Strategy to handle mismatches:
        - 'replace_with_common': Replace mismatched values with the most common value from train_df
        - 'replace_with_nan': Replace mismatched values with NaN
        - 'remove_rows': Remove rows with mismatched values
        
    Returns:
    --------
    pandas.DataFrame:
        A copy of test_df with categorical mismatches fixed
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy of test_df to avoid modifying the original
    fixed_df = test_df.copy()
    
    # Get mismatches
    mismatches = find_categorical_mismatches(train_df, test_df, categorical_columns)
    
    for col, values in mismatches.items():
        if not values:
            continue
            
        if strategy == 'replace_with_common':
            # Get most common value from train_df
            most_common = train_df[col].astype(str).value_counts().idxmax()
            
            # Create a mask for rows with mismatched values
            mask = fixed_df[col].astype(str).isin(values)
            
            # Replace mismatched values with most common value
            fixed_df.loc[mask, col] = most_common
            
            print(f"Column '{col}': Replaced {mask.sum()} mismatched values with '{most_common}'")
            
        elif strategy == 'replace_with_nan':
            # Create a mask for rows with mismatched values
            mask = fixed_df[col].astype(str).isin(values)
            
            # Replace mismatched values with NaN
            fixed_df.loc[mask, col] = np.nan
            
            print(f"Column '{col}': Replaced {mask.sum()} mismatched values with NaN")
            
        elif strategy == 'remove_rows':
            # Create a mask for rows with mismatched values
            mask = fixed_df[col].astype(str).isin(values)
            
            # Remove rows with mismatched values
            fixed_df = fixed_df[~mask]
            
            print(f"Column '{col}': Removed {mask.sum()} rows with mismatched values")
    
    return fixed_df

def diagnose_categorical_encoding(model, train_df, test_df):
    """
    Diagnose potential issues with categorical encoding in LightGBM models.
    
    Parameters:
    -----------
    model : LGBMClassifier or LGBMRegressor
        The LightGBM model that's raising the error
    train_df : pandas.DataFrame
        The training dataframe used to fit the model
    test_df : pandas.DataFrame
        The test dataframe used for prediction
        
    Returns:
    --------
    dict:
        A dictionary with diagnostic information
    """
    import pandas as pd
    
    results = {
        "categorical_columns": [],
        "column_mismatches": {},
        "category_mismatches": {},
        "recommendations": []
    }
    
    # Try to extract categorical features from the model
    try:
        if hasattr(model, 'categorical_feature_') and model.categorical_feature_ is not None:
            results["categorical_columns"] = model.categorical_feature_
        elif hasattr(model, '_Booster') and hasattr(model._Booster, 'params'):
            if 'categorical_feature' in model._Booster.params:
                categorical_str = model._Booster.params['categorical_feature']
                # Convert string representation to list if needed
                if isinstance(categorical_str, str):
                    results["categorical_columns"] = [c.strip() for c in categorical_str.split(',')]
                else:
                    results["categorical_columns"] = categorical_str
    except:
        # Fall back to inferring categorical columns
        for col in train_df.columns:
            if col in test_df.columns:
                if pd.api.types.is_categorical_dtype(train_df[col]) or train_df[col].dtype == 'object':
                    results["categorical_columns"].append(col)
    
    # Check for columns that are in one dataset but not the other
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    
    if missing_in_test:
        results["column_mismatches"]["missing_in_test"] = list(missing_in_test)
        results["recommendations"].append(f"Columns {missing_in_test} are in training data but missing in test data")
        
    if missing_in_train:
        results["column_mismatches"]["missing_in_train"] = list(missing_in_train)
        results["recommendations"].append(f"Columns {missing_in_train} are in test data but missing in training data")
    
    # Check categorical mismatches
    cat_mismatches = find_categorical_mismatches(train_df, test_df, results["categorical_columns"])
    
    if cat_mismatches:
        results["category_mismatches"] = cat_mismatches
        for col, values in cat_mismatches.items():
            results["recommendations"].append(
                f"Column '{col}' has {len(values)} categories in test data that don't exist in training data: {values}"
            )
        
        results["recommendations"].append(
            "Consider using fix_categorical_mismatches() function to handle these mismatches"
        )
    
    return results

# Demo usage:
# train_df and test_df should be your actual dataframes
# model should be your LightGBM model
# 
# Example:
# 
# # Find mismatches
# mismatches = find_categorical_mismatches(train_df, test_df)
# print("Categorical mismatches:", mismatches)
# 
# # Fix mismatches
# fixed_test_df = fix_categorical_mismatches(train_df, test_df, strategy='replace_with_common')
# 
# # Now try prediction with fixed data
# y_pred = model.predict(fixed_test_df)
#
# # For more detailed diagnostics:
# diagnostics = diagnose_categorical_encoding(model, train_df, test_df)
# print(diagnostics)