# train_example.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import numpy as np

# --- Parameters ---
DATA_PATH = 'data/iris_with_categorical.csv' # Assume you have a CSV like this
TARGET_COLUMN = 'species'
MODEL_OUTPUT_DIR = 'models'
MODEL_FILENAME = 'iris_lgbm_classifier.txt' # Use .txt for LGBM Booster save_model
CONFIG_OUTPUT_PATH = 'config.json' # Output path for the config

# --- Create Dummy Data (if needed) ---
def create_dummy_iris_data(path):
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Make species string-based
    le = LabelEncoder()
    df[TARGET_COLUMN] = le.fit_transform(iris.target)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    # Add a dummy categorical feature
    np.random.seed(42)
    df['garden_location'] = np.random.choice(['Sunny', 'Shady', 'Mixed'], size=len(df))
    # Ensure dtypes are appropriate
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype('category') # Good practice for LGBM
    df['garden_location'] = df['garden_location'].astype('category')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dummy data saved to {path}")
    print("Class name mapping (important for config):", dict(enumerate(le.classes_))) # Shows 0:setosa, 1:versicolor, 2:virginica
    return df, list(le.classes_) # Return df and class names

# --- Load Data ---
# create_dummy_iris_data(DATA_PATH) # Uncomment to generate dummy data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Creating dummy data.")
    df, class_names = create_dummy_iris_data(DATA_PATH)
    # Make sure class names are derived correctly if creating dummy data here
    # If loading real data, determine class names from df[TARGET_COLUMN] if it's categorical/object

# Convert target and categorical features if needed (LGBM handles category dtype well)
if pd.api.types.is_object_dtype(df[TARGET_COLUMN]):
     df[TARGET_COLUMN] = df[TARGET_COLUMN].astype('category')
     class_names = df[TARGET_COLUMN].cat.categories.tolist()
     print(f"Detected object target, converted to category. Class names: {class_names}")
elif pd.api.types.is_categorical_dtype(df[TARGET_COLUMN]):
     class_names = df[TARGET_COLUMN].cat.categories.tolist()
     print(f"Target is categorical. Class names: {class_names}")
else:
    class_names = None # Regression or numerical target

# Convert other object columns to category for LGBM internal handling
for col in df.select_dtypes(include=['object']).columns:
    if col != TARGET_COLUMN:
        df[col] = df[col].astype('category')
        print(f"Converted feature '{col}' to category dtype.")


# --- Feature Engineering / Preprocessing (Minimal example) ---
# In a real scenario: handle missing values, scaling (if needed), etc.
features = [col for col in df.columns if col != TARGET_COLUMN]
X = df[features]
y = df[TARGET_COLUMN]

# Check if classification or regression based on target dtype
is_classification = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)

if is_classification:
    # Need numerical labels for LGBM training if target is category/object
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # Get class names in the order LabelEncoder used them
    # This is CRITICAL for the config's class_names list
    class_names_ordered = le.classes_.tolist()
    print(f"Training classifier. Target encoded. Ordered class names: {class_names_ordered}")

else: # Regression
    y_encoded = y
    print("Training regressor.")


# Identify categorical features by name for LGBM
categorical_features = X.select_dtypes(include=['category']).columns.tolist()
print(f"Identified categorical features for LGBM: {categorical_features}")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if is_classification else None)

# --- Model Training ---
if is_classification:
    model = lgb.LGBMClassifier(objective='multiclass', # or 'binary'
                               random_state=42)
    eval_metric = 'multi_logloss' # or 'logloss', 'auc'
else:
    model = lgb.LGBMRegressor(objective='regression_l1', # or 'regression' (L2)
                              random_state=42)
    eval_metric = 'mae' # or 'rmse', 'mape'


print("Starting model training...")
# Use callbacks for early stopping
callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=1)]

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric=eval_metric,
          categorical_feature=categorical_features if categorical_features else 'auto', # Let LGBM handle category dtypes
          callbacks=callbacks)

print("Model training finished.")

# --- Evaluation ---
print("\nEvaluating model...")
y_pred = model.predict(X_test)
if is_classification:
    # For multiclass, predict_proba might be useful too
    # y_pred_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
else:
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error: {mse:.4f}")

# --- Save Model ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
model_save_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
# Save using Booster's save_model method (recommended for LGBM)
model.booster_.save_model(model_save_path)
# Alternatively, save the scikit-learn wrapper object (less portable for just the model)
# import joblib
# joblib.dump(model, model_save_path.replace('.txt', '.pkl'))
print(f"Model saved to {model_save_path}")

# --- Generate Config Template ---
print("\nGenerating configuration template...")
# Use the *original* DataFrame df before encoding y to get correct target type/options
# Make sure df used here has the same features and dtypes as X used in training
from generate_config import dump_config_from_df
dump_config_from_df(df, "Demo Classification", TARGET_COLUMN, output_path='./models/config_template_generated.json')

print("\n--- IMPORTANT NEXT STEPS ---")
print(f"1. Open 'config_template_generated.json'.")
print(f"2. Set 'model_path' to: '{model_save_path}'")
print(f"3. Verify 'model_type' is correct ('{'classifier' if is_classification else 'regressor'}').")
if is_classification:
    print(f"4. **CRITICAL**: Set 'class_names' to the correctly ordered list: {class_names_ordered}")
else:
     print("4. For regressors, 'class_names' is not needed.")
print(f"5. Review all feature types, ranges, and options.")
print(f"6. Save the reviewed file as '{CONFIG_OUTPUT_PATH}'.")
print(f"7. Place '{CONFIG_OUTPUT_PATH}' and the '{MODEL_OUTPUT_DIR}' folder (containing the model) in the same directory as app.py.")