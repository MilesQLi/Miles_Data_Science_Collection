
import pandas as pd
import numpy as np
import re # Useful for more complex patterns if needed
from typing import Optional, List


def remove_noisy_training_samples(
    df: pd.DataFrame,
    target_col: str,
    index_cols: Optional[List[str]] = None,
    z_score_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    remove_feature_outliers: bool = True,
    remove_target_outliers: bool = True
) -> pd.DataFrame:
    """
    Removes noisy samples from a DataFrame based on Z-scores for numeric features
    and the IQR method for the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        index_cols (Optional[List[str]]): A list of column names to be
            treated as indices (and excluded from feature outlier detection).
            Defaults to None, meaning no additional index columns.
        z_score_threshold (float): The Z-score threshold for identifying
            outliers in numeric features. Outliers are samples where any
            feature's Z-score exceeds this value. Defaults to 3.0.
        iqr_multiplier (float): The IQR multiplier for identifying
            outliers in the target column. Outliers are samples where the
            target value is outside Q1 - multiplier*IQR or Q3 + multiplier*IQR.
            Defaults to 1.5.
        remove_feature_outliers (bool): If True, removes outliers based on
            feature Z-scores. Defaults to True.
        remove_target_outliers (bool): If True, removes outliers based on
            target column IQR. Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame with noisy samples removed.

    Raises:
        TypeError: If df is not a pandas DataFrame or index_cols is not a list (if provided).
        ValueError: If target_col or any column in index_cols is not in df.columns.
        ValueError: If z_score_threshold or iqr_multiplier are not positive.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        logging.info("Input DataFrame is empty. Returning an empty DataFrame.")
        return df.copy() # Return a copy to maintain consistency

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")

    if index_cols is None:
        index_cols = [] # Default to an empty list if None
    elif not isinstance(index_cols, list):
        raise TypeError("'index_cols' must be a list of strings or None.")
    else:
        for col in index_cols:
            if col not in df.columns:
                raise ValueError(f"Index column '{col}' not found in DataFrame columns: {df.columns.tolist()}")

    if not (z_score_threshold > 0):
        raise ValueError("z_score_threshold must be positive.")
    if not (iqr_multiplier > 0):
        raise ValueError("iqr_multiplier must be positive.")

    # Work on a copy to avoid modifying the original DataFrame
    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    logging.info(f"Starting outlier removal. Initial rows: {initial_rows}")
    logging.info(f"Parameters: z_score_threshold={z_score_threshold}, iqr_multiplier={iqr_multiplier}, "
                 f"remove_feature_outliers={remove_feature_outliers}, remove_target_outliers={remove_target_outliers}")


    # Initialize an outlier mask (all False initially)
    # Using the DataFrame's index ensures proper alignment
    overall_outlier_mask = pd.Series([False] * initial_rows, index=df_cleaned.index)

    # --- 1. Feature-based Outlier Detection (Z-score) ---
    if remove_feature_outliers:
        cols_to_exclude_for_features = list(set([target_col] + index_cols))
        
        # Select only numeric features, excluding target and index columns
        numeric_features = df_cleaned.drop(columns=cols_to_exclude_for_features, errors='ignore') \
                                     .select_dtypes(include=[np.number])

        if numeric_features.empty:
            logging.warning("No numeric features found for Z-score outlier detection (after excluding target/index cols).")
        else:
            means = numeric_features.mean()
            stds = numeric_features.std()

            # Identify columns with zero or very small standard deviation (constant columns)
            # These columns won't produce meaningful Z-scores and can cause division by zero.
            # We'll only calculate Z-scores for columns with std > a small epsilon.
            valid_std_cols = stds[stds > 1e-9].index # 1e-9 is a small epsilon
            
            if len(valid_std_cols) < len(stds):
                constant_cols = stds[stds <= 1e-9].index.tolist()
                logging.warning(f"Constant or near-constant columns found and excluded from Z-score calculation: {constant_cols}")

            if not valid_std_cols.empty:
                features_for_zscore = numeric_features[valid_std_cols]
                z_scores = np.abs((features_for_zscore - means[valid_std_cols]) / stds[valid_std_cols])
                
                # An outlier if ANY feature's Z-score is too high for that row
                feature_outliers_mask = (z_scores > z_score_threshold).any(axis=1)
                overall_outlier_mask |= feature_outliers_mask
                logging.info(f"Identified {feature_outliers_mask.sum()} potential outliers based on feature Z-scores (threshold > {z_score_threshold}).")
            else:
                logging.warning("No features with non-zero standard deviation found for Z-score calculation.")

    # --- 2. Target-based Outlier Detection (IQR) ---
    if remove_target_outliers:
        if pd.api.types.is_numeric_dtype(df_cleaned[target_col]):
            Q1 = df_cleaned[target_col].quantile(0.25)
            Q3 = df_cleaned[target_col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                # If IQR is 0, it means 50% of the data is concentrated at one value (Q1=Q3).
                # Outliers would be any value not equal to Q1/Q3.
                logging.warning(f"Target column '{target_col}' has an IQR of 0. "
                                "This means at least 50% of its values are identical. "
                                "IQR outlier detection will flag any value different from this common value.")
                # This definition of lower/upper bound will still work correctly.
                # E.g., if target is [5,5,5,5,6,4], Q1=5, Q3=5, IQR=0.
                # lower_bound = 5 - 1.5*0 = 5. upper_bound = 5 + 1.5*0 = 5
                # (val < 5) | (val > 5) will correctly flag 4 and 6.
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            target_outliers_mask = (df_cleaned[target_col] < lower_bound) | \
                                   (df_cleaned[target_col] > upper_bound)
            overall_outlier_mask |= target_outliers_mask
            logging.info(f"Identified {target_outliers_mask.sum()} potential outliers based on target column '{target_col}' IQR (multiplier: {iqr_multiplier}).")
        else:
            logging.warning(f"Target column '{target_col}' is not numeric. Skipping IQR-based outlier detection for target.")

    # --- Filter DataFrame ---
    # Invert the mask to keep non-outliers
    df_filtered = df_cleaned[~overall_outlier_mask]
    rows_removed = initial_rows - len(df_filtered)

    if rows_removed > 0:
        logging.info(f"Total combined outliers removed: {overall_outlier_mask.sum()}.") # Sum of True values in the final mask
        logging.info(f"Total rows removed: {rows_removed} ({rows_removed / initial_rows:.2%} of original data).")
    else:
        logging.info("No outliers were identified based on the current criteria.")
    
    logging.info(f"Final number of rows: {len(df_filtered)}")

    if len(df_filtered) == 0 and initial_rows > 0:
        logging.warning("All rows were identified as outliers and removed. Returning an empty DataFrame.")

    return df_filtered


def clean_and_convert_to_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Cleans a specified column in a DataFrame by removing common non-numeric
    characters (e.g., $, ,, %) and attempts to convert it to a numeric type (float).

    Handles potential errors during conversion by setting problematic values to NaN.
    Modifies the DataFrame column in place.

    Args:
        df: The pandas DataFrame containing the column.
        col: The name (string) of the column to clean and convert.

    Returns:
        The pandas DataFrame with the cleaned and converted column.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
    """
    # --- 1. Input Validation ---
    if col not in df.columns:
        raise ValueError(f"Error: Column '{col}' not found in the DataFrame.")

    # --- 2. Check if Already Numeric ---
    # If the column is already a numeric type, no action is needed.
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"Info: Column '{col}' is already numeric. No cleaning applied.")
        return df

    # --- 3. Cleaning ---
    # Work on the column data. Use .astype(str) temporarily if needed,
    # but .str methods often handle mixed types or NaN correctly.
    cleaned_col = df[col].copy() # Work on a copy initially

    # Only apply string operations if the dtype suggests strings are present
    if pd.api.types.is_string_dtype(cleaned_col) or pd.api.types.is_object_dtype(cleaned_col):
        # Remove leading/trailing whitespace
        cleaned_col = cleaned_col.str.strip()

        # Remove common characters: $, ,, %
        # Using regex=True allows pattern matching (e.g., [$,%])
        # Add any other common symbols you encounter to the character set '[...]'
        # Note: This specific regex removes '$', ',', and '%' characters.
        cleaned_col = cleaned_col.str.replace(r'[$,%]', '', regex=True)

        # Handle cases where cleaning might result in empty strings ''
        # Replace empty strings with NaN so pd.to_numeric treats them as missing
        cleaned_col = cleaned_col.replace('', np.nan)
        # Also handle potential string 'nan' or 'None' if they resulted from earlier steps
        cleaned_col = cleaned_col.replace(['nan', 'NaN', 'None', 'none'], np.nan)

    # --- 4. Conversion ---
    # Attempt to convert the cleaned column to a numeric type (float by default)
    # errors='coerce' is crucial: it turns any values that *still* can't be
    # converted into NaN (Not a Number) instead of raising an error.
    numeric_col = pd.to_numeric(cleaned_col, downcast='float', errors='coerce')

    # --- 5. Update DataFrame and Report ---
    original_dtype = df[col].dtype
    original_nas = df[col].isna().sum()
    df[col] = numeric_col # Assign the converted column back to the DataFrame
    new_nas = df[col].isna().sum()

    print(f"Cleaned column '{col}'. Original dtype: {original_dtype}, New dtype: {df[col].dtype}.")
    if new_nas > original_nas:
        print(f"Info: Introduced {new_nas - original_nas} NaN values during conversion.")
    print(f"Total NaNs in '{col}': {new_nas}")


def drop_columns_with_high_cardinality(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None, exclude: List[str] = [], threshold: int = 100) -> List[str]:
    """
    Drops columns from the DataFrame that have a number of unique values greater than the specified threshold.

    Args:
        df: The pandas DataFrame from which to drop columns.
        threshold: The maximum number of unique values allowed in a column before it is dropped.

    Returns:
        The pandas DataFrame with high cardinality columns removed.
    """
    # Identify columns with high cardinality
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_cols = [col for col in cat_cols if train_df[col].nunique() > threshold and col not in exclude]

    # Drop high cardinality columns
    train_df.drop(columns=high_cardinality_cols, inplace=True)
    if test_df is not None:
        test_df.drop(columns=high_cardinality_cols, inplace=True)

    print(f"Dropped {len(high_cardinality_cols)} columns with cardinality greater than {threshold}.")
    print(f"Dropped columns: {high_cardinality_cols}")
    return high_cardinality_cols