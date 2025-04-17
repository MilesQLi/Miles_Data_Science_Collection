import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, Tuple

def numerical_correlation(df: pd.DataFrame, col1: str, col2: str) -> Tuple[float, str]:
    """
    Calculate correlation between two numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        col1 (str): Name of first numerical column
        col2 (str): Name of second numerical column
    
    Returns:
        Tuple[float, str]: Correlation coefficient and method used
    """
    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("One or both column names not found in DataFrame")
    
    # Check if columns are numerical
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        raise ValueError("Both columns must be numerical")
    
    # Calculate Pearson correlation
    correlation = df[col1].corr(df[col2], method='pearson')
    return correlation, 'pearson'

def numerical_categorical_correlation(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> Tuple[float, str]:
    """
    Calculate correlation between a numerical and categorical column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_col (str): Name of numerical column
        categorical_col (str): Name of categorical column
    
    Returns:
        Tuple[float, str]: Correlation coefficient and method used
    """
    # Check if columns exist
    if numerical_col not in df.columns or categorical_col not in df.columns:
        raise ValueError("One or both column names not found in DataFrame")
    
    # Check if columns are of correct types
    if not pd.api.types.is_numeric_dtype(df[numerical_col]):
        raise ValueError(f"{numerical_col} must be numerical")
    
    # Get unique categories
    n_categories = df[categorical_col].nunique()
    
    if n_categories == 2:
        # For binary categorical variables, use point-biserial correlation
        # Convert categorical to numeric (0, 1)
        categorical_numeric = pd.Categorical(df[categorical_col]).codes
        correlation, _ = stats.pointbiserialr(categorical_numeric, df[numerical_col])
        method = 'point-biserial'
    else:
        # For multi-class categorical variables, use correlation ratio (eta)
        categories = df[categorical_col].unique()
        grand_mean = df[numerical_col].mean()
        n_total = len(df)
        
        # Calculate correlation ratio (η)
        ss_total = ((df[numerical_col] - grand_mean) ** 2).sum()
        ss_between = sum(len(df[df[categorical_col] == cat]) * 
                        (df[df[categorical_col] == cat][numerical_col].mean() - grand_mean) ** 2 
                        for cat in categories)
        
        correlation = np.sqrt(ss_between / ss_total)
        method = 'correlation-ratio'
    
    return correlation, method

def categorical_correlation(df: pd.DataFrame, col1: str, col2: str) -> Tuple[float, str]:
    """
    Calculate correlation between two categorical columns using Cramer's V.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        col1 (str): Name of first categorical column
        col2 (str): Name of second categorical column
    
    Returns:
        Tuple[float, str]: Correlation coefficient and method used
    """
    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("One or both column names not found in DataFrame")
    
    # Create contingency table
    contingency = pd.crosstab(df[col1], df[col2])
    
    # Calculate Chi-square statistic and Cramer's V
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = len(df)
    min_dim = min(contingency.shape) - 1
    
    # Calculate Cramer's V
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    return cramer_v, 'cramer_v'

def get_correlation(df: pd.DataFrame, col1: str, col2: str) -> Tuple[float, str]:
    """
    Automatically determine and calculate the appropriate correlation between two columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        col1 (str): Name of first column
        col2 (str): Name of second column
    
    Returns:
        Tuple[float, str]: Correlation coefficient and method used
    """
    # Check if columns are numerical
    is_num1 = pd.api.types.is_numeric_dtype(df[col1])
    is_num2 = pd.api.types.is_numeric_dtype(df[col2])
    
    if is_num1 and is_num2:
        return numerical_correlation(df, col1, col2)
    elif is_num1 and not is_num2:
        return numerical_categorical_correlation(df, col1, col2)
    elif not is_num1 and is_num2:
        return numerical_categorical_correlation(df, col2, col1)
    else:
        return categorical_correlation(df, col1, col2)
