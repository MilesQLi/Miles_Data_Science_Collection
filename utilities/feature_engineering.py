import pandas as pd

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