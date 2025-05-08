import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_string_dtype


# Set a professional and clean theme globally
sns.set_theme(style="whitegrid", palette="viridis") # "whitegrid", "darkgrid", "ticks", "white"
                                            # Palettes: "pastel", "muted", "bright", "viridis", "Set2", etc.

def plot_one_num_two_cat(df,num,cat,tar):
    sns.displot(data=df, x=num,hue=cat, col=tar,  kde=True)
    return

def plot_three_categorical_correlation(df: pd.DataFrame, fea1: str, fea2: str, target: str):
    """
    Draws the correlation between two categorical features (fea1, fea2) and a categorical target.

    The function creates a faceted bar plot where:
    - Each facet (column of subplots) represents a unique category from `fea1`.
    - Within each facet, the x-axis shows unique categories from `fea2`.
    - The y-axis shows the percentage of each `target` category.
    - Different `target` categories are represented by different colored bars (hue).

    Args:
        df (pd.DataFrame): The input DataFrame.
        fea1 (str): The name of the first categorical feature column.
        fea2 (str): The name of the second categorical feature column.
        target (str): The name of the categorical target column.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the plot.
                                  Returns None if input validation fails.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' must be a pandas DataFrame.")
        return None
    for col in [fea1, fea2, target]:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in DataFrame.")
            return None
        if not pd.api.types.is_categorical_dtype(df[col]) and \
           not pd.api.types.is_object_dtype(df[col]) and \
           not pd.api.types.is_string_dtype(df[col]): # Check if it's reasonably categorical
            print(f"Warning: Column '{col}' might not be categorical. Results may be unexpected.")
        # Convert to category if object or string, for consistent behavior and better performance
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')


    # --- Data Preparation ---
    # Group by fea1, fea2, and then count target categories, then normalize to get proportions
    # Using .size().groupby(level=[0,1]).apply(lambda x: 100 * x / x.sum()) is more robust for this.
    # Or value_counts(normalize=True)
    try:
        # Calculate proportions
        # Group by fea1 and fea2, then get value counts of target, normalized
        grouped = df.groupby([fea1, fea2])[target].value_counts(normalize=True).mul(100)
        plot_data = grouped.rename('percentage').reset_index()

        if plot_data.empty:
            print(f"No data to plot. Check if columns '{fea1}', '{fea2}', '{target}' have overlapping data.")
            return None

    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None

    # --- Plotting ---
    num_fea1_categories = df[fea1].nunique()
    num_fea2_categories = df[fea2].nunique()

    # Adjust figure size based on number of categories to avoid clutter
    # These are heuristic values, might need tweaking
    fig_width = max(8, num_fea1_categories * 3, num_fea2_categories * 0.5 * num_fea1_categories)
    fig_height = max(5, num_fea2_categories * 0.3) if num_fea1_categories > 1 else 5


    plt.figure(figsize=(fig_width, fig_height)) # Set overall figure size for the FacetGrid

    g = sns.catplot(
        data=plot_data,
        x=fea2,
        y='percentage',
        hue=target,
        col=fea1,  # Creates facets for each category in fea1
        kind='bar',
        palette='viridis', # You can choose other palettes like 'muted', 'bright', etc.
        height=max(4, fig_height * 0.8 / (num_fea1_categories if num_fea1_categories > 3 else 1) ), # Height of each facet
        aspect=max(0.8, fig_width / (fig_height * num_fea1_categories) if num_fea1_categories > 0 else 1) # Aspect ratio of each facet
    )

    # --- Customization ---
    g.set_axis_labels(f"{fea2}", "Percentage (%)")
    g.set_titles("{col_var} = {col_name}") # Titles for each subplot (facet)

    # Rotate x-axis labels if they are long or numerous
    if num_fea2_categories > 5 or any(len(str(cat)) > 8 for cat in df[fea2].unique()):
        g.set_xticklabels(rotation=45, ha='right')

    plt.suptitle(f'Distribution of {target} by {fea2}, faceted by {fea1}', y=1.03, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

    # It's good practice to return the figure object if catplot creates a new one
    # or the main axes if a single plot is drawn on existing axes.
    # sns.catplot returns a FacetGrid object, which has a .fig attribute.
    fig = g.fig
    # plt.show() # Call plt.show() outside the function if you want to display it immediately
    return fig

def stacked_plot(df, group, target):
    """
    Function to generate a stacked plots between two variables
    """
    fig, ax = plt.subplots(figsize = (6,4))
    temp_df = (df.groupby([group, target]).size()/df.groupby(group)[target].count()).reset_index().pivot(columns=target, index=group, values=0)
    temp_df.plot(kind='bar', stacked=True, ax = ax, color = ["green", "darkred"])
    ax.xaxis.set_tick_params(rotation=0)
    ax.set_xlabel(group)
    ax.set_ylabel('Churn Percentage')

def display_num_corr(df):
    plt.figure(figsize=(10, 8))
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    numeric_df_for_corr = df[numeric_features].dropna()
    
    corr_matrix = numeric_df_for_corr.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

def display_basic_info(df):
    # --- Display Initial Info ---
    print("======")
    print("\n Number of Rows and Columns:\n", df.shape)
    print("======")
    print("\nData Info:\n")
    df.info()
    print("======")
    print("\nNumber of Unique Values:\n", df.nunique())
    print("======")
    print("\nMissing Values:\n", df.isnull().sum())
    print("======")
    print("\nFirst 5 Rows:\n", df.head())
    print("======")
    print("\nDescriptive Statistics:\n", df.describe())
    print("======")
    print("\nNumber of Duplicated Data:\n", df.duplicated().sum())
    
    return 

def plot_all_num_vs_target(df, target_col, exclude_cols=None, palette="viridis"):
    """
    Generates distribution plots and boxplots for all numerical features against a target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        exclude_cols (list, optional): A list of columns to exclude from plotting. Defaults to None.
        palette (str or list, optional): Color palette to use for plots. Defaults to "viridis".
    """
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return

    num_fields = df.select_dtypes(include=np.number).columns.tolist()

    # Ensure target_col is not in num_fields to plot against itself
    if target_col in num_fields:
        num_fields.remove(target_col)

    if exclude_cols is not None:
        num_fields = [col for col in num_fields if col not in exclude_cols]

    if not num_fields:
        print("No numerical fields to plot (after exclusions).")
        return

    print(f"--- Plotting Numerical Features vs Target ('{target_col}') ---")
    for col in num_fields:
        if col == target_col: # Should be caught by remove above, but as a safeguard
            continue

        plt.figure(figsize=(14, 6)) # Increased figure size

        # 1. Distribution Plot
        plt.subplot(1, 2, 1)
        # Use hue if target is categorical and has few unique values for better distinction
        is_target_categorical_for_hue = df[target_col].nunique() < 10 and df[target_col].dtype in ['object', 'category']
        
        if is_target_categorical_for_hue:
            sns.histplot(data=df, x=col, kde=True, bins=30, hue=target_col, palette=palette, multiple="stack")
            plt.title(f'Distribution of {col} by {target_col}')
        else:
            sns.histplot(df[col], kde=True, bins=30, color=sns.color_palette(palette, 1)[0]) # Use first color of palette
            plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # 2. Box Plot
        plt.subplot(1, 2, 2)
        # Check if target_col is suitable for x-axis in boxplot (categorical or few numerics)
        if df[target_col].nunique() < 20 : # Arbitrary threshold, adjust as needed
            sns.boxplot(x=target_col, y=col, data=df, palette=palette)
            plt.title(f'{col} vs {target_col}')
        else: # If target has too many unique values, a scatter plot might be better
            sns.scatterplot(x=target_col, y=col, data=df, alpha=0.6, color=sns.color_palette(palette, 1)[0])
            plt.title(f'{col} vs Continuous {target_col}')
            
        plt.xlabel(target_col)
        plt.ylabel(col)

        plt.suptitle(f'Analysis of Numerical Feature: {col}', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
        plt.show()
    return

def plot_all_cat_vs_target(df, target_col, exclude_cols=None, palette="Set2"):
    """
    Generates count plots and bar plots (mean of target) for all categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column (should be numeric for the barplot mean).
        exclude_cols (list, optional): A list of columns to exclude from plotting. Defaults to None.
        palette (str or list, optional): Color palette to use for plots. Defaults to "Set2".
    """
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return
    
    # Ensure target_col is numeric for the mean calculation in barplot
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Warning: Target column '{target_col}' is not numeric. Mean calculation in barplot might not be meaningful.")
        # Consider converting if it's binary categorical (e.g., 'Yes'/'No' to 1/0)
        # Or, change the estimator in sns.barplot if appropriate.

    cat_fields = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure target_col is not in cat_fields to plot against itself (if it's categorical)
    if target_col in cat_fields:
        cat_fields.remove(target_col)

    if exclude_cols is not None:
        cat_fields = [col for col in cat_fields if col not in exclude_cols]

    if not cat_fields:
        print("No categorical fields to plot (after exclusions).")
        return

    print(f"--- Plotting Categorical Features vs Target ('{target_col}') ---")
    for col in cat_fields:
        if col == target_col: # Safeguard
            continue

        plt.figure(figsize=(14, 6)) # Increased figure size

        # 1. Count Plot (Distribution of the categorical feature)
        plt.subplot(1, 2, 1)
        # Use hue if target is categorical and has few unique values for better distinction
        is_target_categorical_for_hue = df[target_col].nunique() < 10 and df[target_col].dtype in ['object', 'category']

        if is_target_categorical_for_hue:
             # This shows counts of 'col' categories, colored by 'target_col'
            sns.countplot(x=col, data=df, hue=target_col, palette=palette, order=df[col].value_counts().index[:15]) # Show top 15
        else:
            # This shows counts of 'col' categories, each bar a different color from the palette if desired
            # Or use a single color if preferred. For variety, let's use the palette.
            sns.countplot(x=col, data=df, palette=palette, order=df[col].value_counts().index[:15]) # Show top 15
        
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability

        # 2. Bar Plot (Mean of target_col by category)
        plt.subplot(1, 2, 2)
        # Only makes sense if target_col is numeric or can be meaningfully averaged (e.g., binary 0/1)
        if pd.api.types.is_numeric_dtype(df[target_col]):
            sns.barplot(x=col, y=target_col, data=df, palette=palette, estimator='mean', ci=None, order=df[col].value_counts().index[:15]) # Show top 15, remove confidence interval bars
            plt.title(f'Mean {target_col} by {col}')
        else: # If target is non-numeric, show counts of target by category (less direct than countplot with hue)
            # Or consider a grouped bar chart showing proportions.
            # For simplicity, let's show counts if target is not numeric, though countplot with hue is often better.
            # This plot might become redundant if the target is categorical and used as hue in the countplot.
            try: # Attempt to plot grouped counts if target is categorical
                temp_df = df.groupby(col)[target_col].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
                sns.barplot(x=col, y='percentage', hue=target_col, data=temp_df, palette=palette, order=df[col].value_counts().index[:15])
                plt.title(f'Percentage of {target_col} within {col}')
                plt.ylabel('Percentage')
            except Exception:
                 plt.text(0.5, 0.5, f'{target_col} is non-numeric.\nConsider relationship differently.',
                         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                 plt.title(f'{target_col} by {col}')


        plt.xlabel(col)
        plt.ylabel(f'Mean {target_col}' if pd.api.types.is_numeric_dtype(df[target_col]) else 'Value')
        plt.xticks(rotation=45, ha='right')

        plt.suptitle(f'Analysis of Categorical Feature: {col}', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
        plt.show()
    return

def display_distribution_cat(df,col):
    print(df[col].value_counts(normalize=True))
    plt.figure()
    sns.countplot(x=col,data=df)
    plt.title('Distribution of ' + col)
    return


def get_col_type(series):
  """Identifies if a pandas Series is numerical or categorical."""
  if is_numeric_dtype(series):
    return 'numerical'
  # Consider string columns as categorical for plotting
  elif is_categorical_dtype(series) or is_string_dtype(series):
    return 'categorical'
  else:
    # Handle other types (like datetime, etc.) if needed
    return 'other'

def plot_num_vs_cat(df, num_col, cat_col, plot_type='box'):
    """
    Plots the relationship between a numerical and a categorical column.
    plot_type can be 'box', 'violin', 'bar', 'strip', 'swarm'.
    """
    print(f"Plotting Numerical ({num_col}) vs. Categorical ({cat_col}) using {plot_type} plot")
    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    if plot_type == 'box':
        sns.boxplot(data=df, x=cat_col, y=num_col)
    elif plot_type == 'violin':
        sns.violinplot(data=df, x=cat_col, y=num_col)
    elif plot_type == 'bar':
        # Bar plot shows the mean by default, can add confidence intervals
        sns.barplot(data=df, x=cat_col, y=num_col, errorbar='sd')  # errorbar='sd' for std dev
    elif plot_type == 'strip':
        sns.stripplot(data=df, x=cat_col, y=num_col, alpha=0.7)  # alpha for transparency
    elif plot_type == 'swarm':
        sns.swarmplot(data=df, x=cat_col, y=num_col, size=4)  # size adjusts point size
    else:
        print(f"Plot type '{plot_type}' not recognized. Use 'box', 'violin', 'bar', 'strip', or 'swarm'.")
        return
    plt.title(f'Distribution of {num_col} across {cat_col} categories')
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha='right')  # Rotate labels if many categories
    plt.tight_layout()  # Adjust layout
    plt.show()

def plot_cat_vs_cat(df, col1, col2, normalized=False):
    """
    Plots the relationship between two categorical columns using counts or proportions.
    If normalized=True, shows proportions (percentages); otherwise, shows counts.
    """
    print(f"Plotting Categorical ({col1}) vs. Categorical ({col2})")
    if normalized:
        # Calculate proportions using crosstab
        ct = pd.crosstab(df[col1], df[col2], normalize='index')  # Normalize by row (col1)
        # Or normalize='columns' to normalize by col2, or normalize=True for overall proportion
        print("\nProportions Table:")
        print(ct)
        ct.plot(kind='bar', stacked=True, figsize=(10, 7))
        plt.title(f'Proportion of {col2} within each {col1} category')
        plt.ylabel('Proportion')
    else:
        # Use seaborn's countplot for easy grouped counts
        plt.figure(figsize=(10, 7))
        sns.countplot(data=df, x=col1, hue=col2)
        plt.title(f'Count of {col2} for each {col1} category')
        plt.ylabel('Count')
    plt.xlabel(col1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=col2)
    plt.tight_layout()
    plt.show()
# --- Example Usage ---
# Assuming 'df', 'cat_col1', 'cat_col2'
# plot_cat_vs_cat(df, 'Department', 'Smoker', normalized=False) # Show counts
# plot_cat_vs_cat(df, 'Department', 'Smoker', normalized=True)  # Show proportions
def plot_cat_vs_cat(df, col1, col2, normalized=False):
    """
    Plots the relationship between two categorical columns using counts or proportions.
    If normalized=True, shows proportions (percentages); otherwise, shows counts.
    """
    print(f"Plotting Categorical ({col1}) vs. Categorical ({col2})")
    # Create a copy of the DataFrame with converted columns
    plot_df = df.copy()
    # Convert columns to string to ensure they're treated as categorical
    plot_df[col1] = plot_df[col1].astype(str)
    plot_df[col2] = plot_df[col2].astype(str)
    if normalized:
        # Calculate proportions using crosstab
        ct = pd.crosstab(plot_df[col1], plot_df[col2], normalize='index')
        print("\nProportions Table:")
        print(ct)
        ct.plot(kind='bar', stacked=True, figsize=(10, 7))
        plt.title(f'Proportion of {col2} within each {col1} category')
        plt.ylabel('Proportion')
    else:
        # Use seaborn's countplot for easy grouped counts
        plt.figure(figsize=(10, 7))
        sns.countplot(data=plot_df, x=col1, hue=col2)
        plt.title(f'Count of {col2} for each {col1} category')
        plt.ylabel('Count')
    plt.xlabel(col1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=col2)
    plt.tight_layout()
    plt.show()
# --- Example Usage ---
# Assuming 'df', 'cat_col1', 'cat_col2'
# plot_cat_vs_cat(df, 'Department', 'Smoker', normalized=False) # Show counts
# plot_cat_vs_cat(df, 'Department', 'Smoker', normalized=True)  # Show proportions
def plot_cat_vs_cat_heatmap(df, col1, col2, normalized=False):
    """Plots a heatmap for two categorical columns."""
    print(f"Plotting Heatmap for Categorical ({col1}) vs. Categorical ({col2})")
    ct = pd.crosstab(df[col1], df[col2], normalize=normalized)
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, cmap="Blues", fmt=".2f" if normalized else "d")  # Format as float or integer
    plt.title(f'Relationship between {col1} and {col2} ({"Proportions" if normalized else "Counts"})')
    plt.show()
# plot_cat_vs_cat_heatmap(df, 'Department', 'Smoker', normalized=True)