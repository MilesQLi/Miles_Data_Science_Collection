# generate_config.py
import pandas as pd
import json
import numpy as np
import argparse
import os

def dump_config_from_df(df: pd.DataFrame, project_title: str, target_name: str, model_path: str = 'model.txt', output_path: str = 'config_template.json'):
    """
    Generates a configuration file structure based on a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame (ideally the training data).
        target_name (str): The name of the target column in the DataFrame.
        output_path (str): Path to save the generated JSON configuration template.
    """
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in DataFrame.")

    config = {
        "project_title":  project_title,
        "model_type": "", # To be filled: "classifier" or "regressor"
        "model_path": model_path, # To be filled after training
        "target_name": target_name,
        "target_type": "", # To be filled based on target dtype
        "class_names": [], # To be filled for classifiers if needed
        "features": []
    }

    # Determine target type and model type hint
    target_dtype = df[target_name].dtype
    if pd.api.types.is_numeric_dtype(target_dtype):
        config["target_type"] = "numerical"
        config["model_type"] = "regressor" # Suggestion
    elif pd.api.types.is_categorical_dtype(target_dtype) or pd.api.types.is_object_dtype(target_dtype):
        config["target_type"] = "categorical"
        config["model_type"] = "classifier" # Suggestion
        # Attempt to get class names if categorical, otherwise leave empty
        if pd.api.types.is_categorical_dtype(target_dtype):
             config["class_names"] = df[target_name].cat.categories.tolist()
        else:
             # For object type, sort unique values to have a consistent order
             # IMPORTANT: Ensure this order matches model's internal encoding during training!
             unique_values = sorted(df[target_name].unique().tolist())
             config["class_names"] = [str(v) for v in unique_values] # Ensure string representation
             print(f"WARNING: Target '{target_name}' is object type. "
                   f"Inferred class names: {config['class_names']}. "
                   "Verify this order matches the trained model's class encoding.")

    else:
        config["target_type"] = "unknown"
        config["model_type"] = "unknown" # Needs manual setting

    # Process features
    feature_names = [col for col in df.columns if col != target_name]
    for feature in feature_names:
        feature_info = {"name": feature}
        col_dtype = df[feature].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            feature_info["type"] = "numerical"
            # Calculate range, handle potential NaNs
            min_val = df[feature].min()
            max_val = df[feature].max()
            feature_info["range"] = [
                float(min_val) if pd.notna(min_val) else None,
                float(max_val) if pd.notna(max_val) else None
            ]
            feature_info["dtype"] = str(col_dtype)
        elif pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
            feature_info["type"] = "categorical"
            unique_options = df[feature].unique()
            # Ensure options are JSON serializable (strings) and handle potential NaNs
            feature_info["options"] = sorted([str(opt) for opt in unique_options if pd.notna(opt)])
            feature_info["dtype"] = "category" if pd.api.types.is_categorical_dtype(col_dtype) else "object"
        else:
            feature_info["type"] = "unknown" # Needs manual review
            feature_info["range"] = None
            feature_info["options"] = None
            feature_info["dtype"] = str(col_dtype)

        config["features"].append(feature_info)

    # Save the config template
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration template saved to '{output_path}'.")
    print("Please review and fill in 'model_type', 'model_path', and verify 'class_names' (if applicable).")

# Example Usage (replace with your actual data loading)
if __name__ == "__main__":
    # Create a dummy DataFrame for demonstration
    data = {
        'sepal length (cm)': [5.1, 4.9, 4.7, 7.0, 6.4, 6.9],
        'sepal width (cm)': [3.5, 3.0, 3.2, 3.2, 3.2, 3.1],
        'petal length (cm)': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9],
        'petal width (cm)': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5],
        'garden_location': ['Sunny', 'Shady', 'Sunny', 'Mixed', 'Shady', 'Mixed'],
        'species': ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor'] # Target
    }
    dummy_df = pd.DataFrame(data)
    # Convert species to categorical for better handling (optional but good practice)
    dummy_df['species'] = pd.Categorical(dummy_df['species'])
    dummy_df['garden_location'] = pd.Categorical(dummy_df['garden_location'])


    parser = argparse.ArgumentParser(description="Generate config template from DataFrame.")
    # In a real script, you'd load df from a file, e.g., CSV
    # parser.add_argument("data_path", help="Path to the data file (e.g., CSV)")
    parser.add_argument("target_column", help="Name of the target column")
    parser.add_argument("-o", "--output", default="config_template.json", help="Output config file path")

    args = parser.parse_args()

    # Load your actual DataFrame here, e.g., from args.data_path
    # Example: actual_df = pd.read_csv(args.data_path)
    actual_df = dummy_df # Using dummy data for this example

    dump_config_from_df(actual_df, args.target_column, args.output)

    # Command to run:
    # python generate_config.py species -o config.json