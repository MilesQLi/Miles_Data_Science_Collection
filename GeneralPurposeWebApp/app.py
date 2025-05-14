# app.py
import flask
import lightgbm as lgb
import pandas as pd
import json
import os
import numpy as np
# import joblib # Use joblib if your model is saved as .pkl

# --- Configuration ---
CONFIG_FILE = os.path.dirname(os.path.abspath(__file__))+'/config.json'
MODEL = None
CONFIG = None

# --- Helper Functions ---
def load_config(config_path):
    """Loads the configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print("Configuration loaded successfully.")
        # Basic validation
        required_keys = ["model_type", "model_path", "target_name", "features"]
        if not all(key in config_data for key in required_keys):
            raise ValueError("Config file missing required keys.")
        if not isinstance(config_data["features"], list):
             raise ValueError("'features' must be a list in config.")
        if config_data["model_type"] not in ["classifier", "regressor"]:
            raise ValueError("model_type must be 'classifier' or 'regressor'.")
        if config_data["model_type"] == "classifier" and "class_names" not in config_data:
             print("WARNING: model_type is 'classifier' but 'class_names' not found in config. Prediction will be index/probability.")

        return config_data
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {config_path}")
        return None
    except ValueError as e:
        print(f"ERROR: Invalid configuration: {e}")
        return None

def load_model(config):
    """Loads the LightGBM model based on the configuration."""
    if not config or 'model_path' not in config:
        return None
    model_path = config['model_path']
    try:
        # Check file extension to decide loading method (optional)
        if model_path.lower().endswith('.txt'):
            bst = lgb.Booster(model_file=model_path)
            print(f"LightGBM model loaded successfully from {model_path} (.txt format).")
        # Example for .pkl (if you save using joblib/pickle)
        # elif model_path.lower().endswith('.pkl'):
        #     bst = joblib.load(model_path)
        #     print(f"Model loaded successfully from {model_path} (.pkl format).")
        else:
             # Assume .txt if extension unknown or different
             print(f"WARNING: Unknown model file extension for {model_path}. Assuming LightGBM Booster (.txt) format.")
             bst = lgb.Booster(model_file=model_path)

        # Validate model features if possible (LightGBM Booster API has feature_name())
        if hasattr(bst, 'feature_name'):
            model_features = bst.feature_name()
            config_features = [f['name'] for f in config['features']]
            if set(model_features) != set(config_features):
                 print("WARNING: Features in loaded model do not exactly match features in config file!")
                 print(f"  Model features: {model_features}")
                 print(f"  Config features: {config_features}")
                 # Decide if this is critical - maybe only warn or raise error
                 # raise ValueError("Feature mismatch between model and config.")
            else:
                 print("Model features match config features.")
        else:
            print("Could not verify model features against config (model object type might not support it).")

        return bst
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return None
    except lgb.basic.LightGBMError as e:
         print(f"ERROR: Failed to load LightGBM model from {model_path}. Error: {e}")
         return None
    # except Exception as e: # Catch other potential errors like pickle errors
    #     print(f"ERROR: An unexpected error occurred loading model from {model_path}: {e}")
    #     return None

# --- Flask App Initialization ---
app = flask.Flask(__name__)

# --- Load Config and Model at Startup ---
CONFIG = load_config(CONFIG_FILE)
if CONFIG:
    MODEL = load_model(CONFIG)
else:
    print("CRITICAL: Could not load configuration. The application might not work correctly.")
    # Optionally exit or prevent app run if config is essential
    # sys.exit(1)

if CONFIG and not MODEL:
     print("CRITICAL: Configuration loaded, but failed to load the model. Check model path and file integrity.")
     # Optionally exit or prevent app run
     # sys.exit(1)


# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if not CONFIG or not MODEL:
        return flask.render_template('error.html',
                                     error_message="Application initialization failed. Check logs.")

    prediction_result = None
    prediction_input = None
    error_message = None

    if flask.request.method == 'POST':
        try:
            # 1. Get data from form
            input_data = {}
            feature_order = [] # Keep track of order defined in config
            dtypes = {}      # Store expected dtypes

            for feature in CONFIG['features']:
                feature_name = feature['name']
                feature_order.append(feature_name)
                dtypes[feature_name] = feature.get('dtype', 'object') # Default to object if dtype missing

                value = flask.request.form.get(feature_name)
                if value is None or value == '':
                    raise ValueError(f"Missing value for required feature: {feature_name}")

                # 2. Basic type conversion and validation
                if feature['type'] == 'numerical':
                    try:
                        # Convert numerical features to float (LGBM generally prefers float)
                        input_data[feature_name] = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid numerical value '{value}' for feature: {feature_name}")
                elif feature['type'] == 'categorical':
                    # Keep categorical features as strings for now.
                    # LGBM can handle strings if trained with categorical_feature parameter
                    # OR if data was pre-encoded (which needs to be handled carefully).
                    # Ensure the value is one of the allowed options.
                    if 'options' in feature and value not in feature['options']:
                         raise ValueError(f"Invalid option '{value}' for categorical feature: {feature_name}. Allowed: {feature['options']}")
                    input_data[feature_name] = str(value) # Ensure string type
                else:
                    # Handle unknown types if necessary, or raise error
                     input_data[feature_name] = value # Pass as is

            prediction_input = input_data.copy() # Save user's input for re-display

            # 3. Prepare DataFrame for prediction
            # Create DataFrame with a single row
            # Important: Column order should ideally match training data if model is sensitive,
            # but LGBM's predict method usually handles order if column names are correct.
            # Explicitly use feature_order derived from config for consistency.
            input_df = pd.DataFrame([input_data], columns=feature_order)

            # 4. Ensure correct dtypes (crucial for some models/pipelines)
            # Convert columns to the dtypes specified in config if possible
            for feature_name, dtype_str in dtypes.items():
                try:
                    if dtype_str == 'category':
                         # If the model expects pandas Categorical type:
                         # Need the original categories from training! Storing them in config might be needed.
                         # For simplicity here, we'll keep as object/string and rely on LGBM handling or pre-encoding.
                         # If using pd.Categorical, ensure categories match training data:
                         # categories = CONFIG['features'][feature_order.index(feature_name)].get('options', []) # Get options as categories
                         # input_df[feature_name] = pd.Categorical(input_df[feature_name], categories=categories)
                         input_df[feature_name] = input_df[feature_name].astype(object) # Keep as object for now
                         print(f"DEBUG: Feature '{feature_name}' kept as object (LGBM might handle internally if specified during training).")
                    elif dtype_str: # Check if dtype_str is not empty
                        input_df[feature_name] = input_df[feature_name].astype(dtype_str)

                except Exception as e:
                    print(f"Warning: Could not convert column '{feature_name}' to specified dtype '{dtype_str}'. Error: {e}. Using default dtype.")
                    # Fallback or raise error depending on strictness needed
            
            cat_cols = input_df.select_dtypes(include='object').columns.tolist()
            for col in cat_cols:
                input_df[col] = input_df[col].astype('category')


            print(f"DEBUG: Input DataFrame for prediction:\n{input_df.info()}\n==========")
            print(input_df)
            print("==========")

            # 5. Make prediction
            prediction = MODEL.predict(input_df) # Pass the DataFrame
            print(f"DEBUG: Raw prediction output: {prediction}")

            # 6. Format prediction result
            if CONFIG['model_type'] == 'classifier':
                # Prediction might be class probabilities or class index
                if prediction.ndim > 1 and prediction.shape[1] > 1: # Probabilities per class
                    predicted_class_index = np.argmax(prediction[0])
                    probability = prediction[0][predicted_class_index]
                    if 'class_names' in CONFIG and len(CONFIG['class_names']) > predicted_class_index:
                        predicted_class_name = CONFIG['class_names'][predicted_class_index]
                        prediction_result = f"Predicted Class: {predicted_class_name} (Probability: {probability:.4f})"
                    else: # Fallback if class names aren't available/correct
                        prediction_result = f"Predicted Class Index: {predicted_class_index} (Probability: {probability:.4f})"
                else: # Single prediction value (likely class index)
                     predicted_class_index = int(prediction[0])
                     if 'class_names' in CONFIG and len(CONFIG['class_names']) > predicted_class_index:
                         predicted_class_name = CONFIG['class_names'][predicted_class_index]
                         prediction_result = f"Predicted Class: {predicted_class_name}"
                     else: # Fallback if class names aren't available/correct
                         prediction_result = f"Predicted Class Index: {predicted_class_index}"

            elif CONFIG['model_type'] == 'regressor':
                # Prediction is a single numerical value
                predicted_value = prediction[0]
                prediction_result = f"Predicted Value: {predicted_value:.4f}" # Format as needed

        except ValueError as e:
            error_message = f"Input Error: {e}"
        except lgb.basic.LightGBMError as e:
             error_message = f"Prediction Error: {e}. Ensure input data matches model expectations."
             print(f"ERROR during prediction: {e}")
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"ERROR during request processing: {e}")
            import traceback
            traceback.print_exc() # Print full traceback to console for debugging

    # Render the form (either initially or after POST)
    return flask.render_template('index.html',
                                 config=CONFIG,
                                 prediction_result=prediction_result,
                                 prediction_input=prediction_input, # Pass back input to repopulate form
                                 error_message=error_message,
                                 project_title=CONFIG.get('project_title', 'General Purpose Predictor'))

@app.route('/config')
def show_config():
     """A simple route to display the loaded configuration for debugging."""
     if not CONFIG:
          return "Configuration not loaded.", 500
     return flask.jsonify(CONFIG)

# --- Run the App ---
if __name__ == '__main__':
    # Set debug=True for development (auto-reloads, detailed errors)
    # Set host='0.0.0.0' to make it accessible on your network
    # Set use_reloader=False if you encounter issues with model loading on reload
    app.run(debug=True, host='0.0.0.0', port=5000)