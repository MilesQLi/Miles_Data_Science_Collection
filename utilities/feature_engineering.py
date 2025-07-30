import pandas as pd
from typing import List
from tqdm import tqdm

def cluster_and_calculate_means(
    df: pd.DataFrame,
    clustering_features: list[str],
    n_clusters: int,
    mean_features: list[str],
    cluster_label_col: str = 'cluster_label',
    scale_clustering_features: bool = True,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Clusters samples in a DataFrame using K-Means based on specified features,
    calculates the mean of other specified features for each cluster, and
    assigns the cluster label and calculated means back to the original samples.

    Args:
        df (pd.DataFrame): The input DataFrame containing the samples and features.
        clustering_features (list[str]): A list of column names in 'df' to be
                                         used for clustering.
        n_clusters (int): The number of clusters to form (k in K-Means).
        mean_features (list[str]): A list of column names in 'df' for which
                                   the mean should be calculated within each cluster.
        cluster_label_col (str, optional): The name for the new column that will
                                           store the cluster assignment for each sample.
                                           Defaults to 'cluster_label'.
        scale_clustering_features (bool, optional): Whether to standardize
                                                    (mean=0, variance=1) the
                                                    clustering features before
                                                    running K-Means. Highly recommended
                                                    if features have different scales.
                                                    Defaults to True.
        random_state (int | None, optional): Determines random number generation for
                                             centroid initialization. Use an int for
                                             reproducible results. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus two types
                      of added columns:
                      1. The cluster assignment ('cluster_label' by default).
                      2. The mean of each feature in 'mean_features' calculated
                         for the assigned cluster (e.g., 'feature_A_cluster_mean').

    Raises:
        ValueError: If any specified feature columns are not in the DataFrame,
                    if n_clusters is not positive, or if input types are incorrect.
        ImportError: If pandas or scikit-learn is not installed.
    """
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import warnings

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(clustering_features, list) or not all(isinstance(f, str) for f in clustering_features):
        raise ValueError("'clustering_features' must be a list of strings.")
    if not isinstance(mean_features, list) or not all(isinstance(f, str) for f in mean_features):
        raise ValueError("'mean_features' must be a list of strings.")
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("'n_clusters' must be a positive integer.")
    if not isinstance(cluster_label_col, str):
        raise ValueError("'cluster_label_col' must be a string.")

    all_features = set(clustering_features + mean_features)
    missing_cols = all_features - set(df.columns)
    if missing_cols:
        raise ValueError(f"The following specified columns are missing from the DataFrame: {missing_cols}")

    # Check for NaNs in clustering features, as K-Means doesn't handle them
    if df[clustering_features].isnull().any().any():
        warnings.warn(
            f"NaN values found in clustering features: {df[clustering_features].isnull().sum().loc[lambda x: x>0].index.tolist()}. "
            "K-Means may fail or produce unexpected results. Consider dropping or imputing NaNs before calling this function.",
            UserWarning
        )
        # Or raise ValueError("NaN values found in clustering features. Please handle them before clustering.")

    # --- Prepare Data for Clustering ---
    df_result = df.copy() # Work on a copy to avoid modifying the original df
    X_cluster = df_result[clustering_features].values

    # --- Optional Scaling ---
    if scale_clustering_features:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(X_cluster)
        # Note: We fit K-Means on scaled data, but calculate means on original data

    # --- Perform Clustering ---
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto' # Suppresses future warning, runs multiple initializations
    )
    cluster_labels = kmeans.fit_predict(X_cluster)

    # --- Assign Cluster Labels ---
    df_result[cluster_label_col] = cluster_labels

    # --- Calculate Cluster Means ---
    # Group by the newly assigned cluster label and calculate mean for specified features
    cluster_means = df_result.groupby(cluster_label_col)[mean_features].mean()

    # Rename the mean columns to avoid conflicts and clarify their meaning
    mean_col_mapping = {col: f"{col}_cluster_mean" for col in mean_features}
    cluster_means = cluster_means.rename(columns=mean_col_mapping)

    # --- Assign Cluster Means back to Samples ---
    # Merge the calculated means back into the main DataFrame based on cluster label
    # We merge on the cluster label column in df_result and the index (which is the cluster label) in cluster_means
    df_result = df_result.merge(
        cluster_means,
        left_on=cluster_label_col,
        right_index=True,
        how='left' # Use left join to keep all original rows
    )

    return df_result




def cluster_and_assign_group_prediction(
    df: pd.DataFrame,
    num_training_samples: int,
    clustering_features: list[str],
    n_clusters: int,
    target_col: str,
    cluster_label_col: str = 'cluster_label',
    scale_clustering_features: bool = True,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Clusters samples in a DataFrame using K-Means based on specified features,
    calculates the mean of other specified features for each cluster, and
    assigns the cluster label and calculated means back to the original samples.

    Args:
        df (pd.DataFrame): The input DataFrame containing the samples and features.
        clustering_features (list[str]): A list of column names in 'df' to be
                                         used for clustering.
        n_clusters (int): The number of clusters to form (k in K-Means).
        mean_features (list[str]): A list of column names in 'df' for which
                                   the mean should be calculated within each cluster.
        cluster_label_col (str, optional): The name for the new column that will
                                           store the cluster assignment for each sample.
                                           Defaults to 'cluster_label'.
        scale_clustering_features (bool, optional): Whether to standardize
                                                    (mean=0, variance=1) the
                                                    clustering features before
                                                    running K-Means. Highly recommended
                                                    if features have different scales.
                                                    Defaults to True.
        random_state (int | None, optional): Determines random number generation for
                                             centroid initialization. Use an int for
                                             reproducible results. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus two types
                      of added columns:
                      1. The cluster assignment ('cluster_label' by default).
                      2. The mean of each feature in 'mean_features' calculated
                         for the assigned cluster (e.g., 'feature_A_cluster_mean').

    Raises:
        ValueError: If any specified feature columns are not in the DataFrame,
                    if n_clusters is not positive, or if input types are incorrect.
        ImportError: If pandas or scikit-learn is not installed.
    """
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import warnings

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(num_training_samples, int) or num_training_samples <= 0:
        raise ValueError("'num_training_samples' must be a positive integer.")
    if not isinstance(clustering_features, list) or not all(isinstance(f, str) for f in clustering_features):
        raise ValueError("'clustering_features' must be a list of strings.")
    if not isinstance(target_col, str):
        raise ValueError("'target_col' must be a string.")
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("'n_clusters' must be a positive integer.")
    if not isinstance(cluster_label_col, str):
        raise ValueError("'cluster_label_col' must be a string.")


    # Check for NaNs in clustering features, as K-Means doesn't handle them
    if df[clustering_features].isnull().any().any():
        warnings.warn(
            f"NaN values found in clustering features: {df[clustering_features].isnull().sum().loc[lambda x: x>0].index.tolist()}. "
            "K-Means may fail or produce unexpected results. Consider dropping or imputing NaNs before calling this function.",
            UserWarning
        )
        # Or raise ValueError("NaN values found in clustering features. Please handle them before clustering.")

    # --- Prepare Data for Clustering ---
    df_result = df.copy() # Work on a copy to avoid modifying the original df
    X_cluster = df_result[clustering_features]
    cat_cols = df_result.select_dtypes(include=['object']).columns.tolist()
    X_cluster = pd.get_dummies(X_cluster, columns=cat_cols, drop_first=True)
    X_cluster = X_cluster.values

    # --- Optional Scaling ---
    if scale_clustering_features:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(X_cluster)
        # Note: We fit K-Means on scaled data, but calculate means on original data

    # --- Perform Clustering ---
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto' # Suppresses future warning, runs multiple initializations
    )
    cluster_labels = kmeans.fit_predict(X_cluster)

    # --- Assign Cluster Labels ---
    df_result[cluster_label_col] = cluster_labels

    # --- Calculate Cluster Means ---
    # Group by the newly assigned cluster label and calculate mean for specified features
    cluster_means = df_result.iloc[:num_training_samples,:].groupby(cluster_label_col)[target_col].mean()

    df_result['mean_target'] = df_result[cluster_label_col].map(cluster_means)

    return df_result



def target_encode_multiclass(
    df: pd.DataFrame, 
    features: List[str], 
    target: str
) -> pd.DataFrame:
    """
    Performs target encoding for a multi-class target variable.

    For each categorical feature, it calculates the probability of each target class
    and creates new columns with these probabilities. The original categorical 
    features are then dropped.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (List[str]): A list of categorical column names to be encoded.
        target (str): The name of the target column, which should contain class strings.

    Returns:
        pd.DataFrame: A DataFrame with the original features replaced by their
                      target-encoded counterparts.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_encoded = df.copy()

    # Get the unique classes from the target column
    target_classes = df[target].unique()
    
    # 1. One-Hot Encode the Target column to calculate probabilities easily
    # This creates a temporary dataframe with columns for each class (e.g., target_cat, target_dog)
    # The values are 1 if the row corresponds to that class, 0 otherwise.
    one_hot_target = pd.get_dummies(df[target], prefix=target)

    # Combine the one-hot encoded target with the features for grouping
    temp_df = pd.concat([df[features], one_hot_target], axis=1)

    # 2. Iterate through each feature to encode
    for feature in tqdm(features, desc="Encoding Features"):
        
        # 3. Calculate the mean of each target class for each category in the feature
        # This gives us the conditional probability P(target_class | feature_category)
        # The result is a mapping from each feature's category to the class probabilities.
        encoding_map = temp_df.groupby(feature)[one_hot_target.columns].mean()
        
        # 4. Create new, descriptive column names for the encoded features
        # e.g., 'City' -> 'City_target_cat_encoded', 'City_target_dog_encoded'
        new_col_names = {col: f"{feature}_{col}_encoded" for col in encoding_map.columns}
        encoding_map.rename(columns=new_col_names, inplace=True)
        
        # 5. Merge the new encoded columns back into the main dataframe
        # We use a left merge to ensure we keep all original rows
        df_encoded = pd.merge(
            df_encoded, 
            encoding_map,
            how='left',
            left_on=feature,
            right_index=True # Merge on the index of encoding_map (which is the feature's categories)
        )
        
    # 6. Drop the original categorical features that have now been encoded
    df_encoded.drop(columns=features, inplace=True)

    return df_encoded

def consolidate_feature_values_by_abs_shap( # Renamed for clarity
    data: pd.DataFrame,
    shape_values: pd.DataFrame,
    selected_feature_name: str,
    threshold: float
) -> pd.DataFrame:
    """
    Replaces values of a selected categorical feature with "other" if their
    maximum *absolute* SHAP value (for that feature) is below a given threshold.

    Args:
        data (pd.DataFrame): The input dataframe containing the features.
        shape_values (pd.DataFrame): DataFrame of SHAP values.
                                     Rows must correspond to samples in `data`.
                                     Columns must be feature names, including
                                     `selected_feature_name`.
        selected_feature_name (str): The name of the categorical feature column
                                     in `data` to be processed.
        threshold (float): The SHAP value threshold. If the maximum *absolute*
                           SHAP value for a specific category of
                           `selected_feature_name` is less than this threshold,
                           that category will be replaced by "other".
                           The threshold itself should be a positive value.

    Returns:
        pd.DataFrame: A new DataFrame with the `selected_feature_name` column updated.
                      The original `data` DataFrame is not modified.

    Raises:
        ValueError: If `selected_feature_name` is not in `data` or `shape_values`.
        ValueError: If `data` and `shape_values` do not have the same number of rows.
        ValueError: If threshold is negative.
    """
    # --- Input Validations ---
    if selected_feature_name not in data.columns:
        raise ValueError(f"Feature '{selected_feature_name}' not found in data DataFrame columns.")
    if selected_feature_name not in shape_values.columns:
        raise ValueError(f"Feature '{selected_feature_name}' not found in shape_values DataFrame columns.")
    if len(data) != len(shape_values):
        raise ValueError(
            f"Data ({len(data)} rows) and shape_values ({len(shape_values)} rows) "
            "must have the same number of samples."
        )
    if threshold < 0:
        raise ValueError("Threshold for absolute SHAP values must be non-negative.")

    # Work on a copy to avoid modifying the original DataFrame
    data_updated = data.copy()

    # Ensure consistent indexing for alignment
    temp_shape_values = shape_values.reset_index(drop=True)
    temp_data_feature_column = data_updated[selected_feature_name].reset_index(drop=True)

    # Extract SHAP values specifically for the selected feature
    shap_for_selected_feature = temp_shape_values[selected_feature_name]

    # Create a temporary DataFrame to easily group feature values with their SHAP values
    analysis_df = pd.DataFrame({
        'feature_value': temp_data_feature_column,
        'shap_value': shap_for_selected_feature
    })

    # For each unique value in the selected feature, find its maximum *absolute* SHAP value
    # This applies .abs() to the 'shap_value' Series within each group,
    # then finds the maximum of those absolute values.
    max_abs_shap_per_category = analysis_df.groupby('feature_value')['shap_value'].apply(lambda x: x.abs().max())
    # Or, as you correctly suggested, which is more concise and often more performant for built-in functions:
    # max_abs_shap_per_category = analysis_df.groupby('feature_value')['shap_value'].abs().max()
    # Let's stick to your concise version as it's more direct for this operation.
    #max_abs_shap_per_category = analysis_df.groupby('feature_value')['shap_value'].abs().max()


    # Identify categories where their maximum absolute SHAP value is less than the threshold
    categories_to_replace = max_abs_shap_per_category[max_abs_shap_per_category < threshold].index.tolist()

    if "other" in categories_to_replace:
        print(f"Warning: The 'other' category itself met the criteria for replacement "
              f"(max absolute SHAP < {threshold}). It will remain 'other'.")

    # Replace these categories with "other" in the updated data
    mask_to_replace = data_updated[selected_feature_name].isin(categories_to_replace)
    data_updated.loc[mask_to_replace, selected_feature_name] = "other"

    return data_updated