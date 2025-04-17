import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_string_dtype

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


