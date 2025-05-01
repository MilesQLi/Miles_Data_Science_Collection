
import pandas as pd
import numpy as np
import re # Useful for more complex patterns if needed
from typing import Optional, List

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