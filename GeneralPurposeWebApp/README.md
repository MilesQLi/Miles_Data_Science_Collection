# General Purpose ML Predictor Web App (Flask + LightGBM)

This project provides a general-purpose Flask web application for serving simple classification or regression models trained using LightGBM (`LGBMClassifier` or `LGBMRegressor`).

The key feature is that the application is driven by a configuration file (`config.json`) which defines the model to use, the expected input features (names, types, valid ranges/options), and target variable details. This allows deploying different LightGBM models with varying features without changing the core application code.

## Features

*   **Web Interface:** Simple HTML form generated dynamically based on the configuration.
*   **Configuration Driven:** Model path, feature definitions (name, type, constraints), target info are all loaded from `config.json`.
*   **Supports LightGBM:** Specifically designed for models saved using LightGBM's `Booster.save_model()` method (typically `.txt` files).
*   **Classification & Regression:** Handles both `LGBMClassifier` and `LGBMRegressor` based on the configuration.
*   **Dynamic UI Elements:**
    *   Dropdowns (`<select>`) for categorical features based on options in the config.
    *   Number inputs (`<input type="number">`) for numerical features with min/max hints from the config.
*   **Input Validation:** Basic checks for missing values and valid options for categorical features.
*   **Prediction Display:** Shows the prediction result (class name/index/probability for classifiers, value for regressors) on the web page.
*   **Configuration Generation Helper:** Includes a script (`generate_config.py`) to create a configuration template from a Pandas DataFrame.
*   **Example Training Script:** Provides `train_example.py` demonstrating how to train a model, save it, and use the config generator.

## Project Structure

```
.
├── models/                  # Directory to store trained model files
│   └── iris_lgbm_classifier.txt  # Example trained model
├── templates/               # Flask HTML templates
│   ├── index.html           # Main prediction form template
│   └── error.html           # Template for application errors
├── app.py                   # The main Flask application logic
├── config.json              # Configuration file driving the app (NEEDS MANUAL EDITING)
├── generate_config.py       # Helper script to generate config template from data
├── train_example.py         # Example script to train a model and generate config template
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
    Or download and extract the source code.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration (`config.json`)

This is the **most important file** for customizing the application. You need to create/edit `config.json` in the root directory.

*   **Generation:** You can generate a template using the `generate_config.py` script after preparing your training data in a Pandas DataFrame:
    ```bash
    # Example: Generate template from 'my_data.csv' where 'target_var' is the target
    # python generate_config.py target_var --data_path my_data.csv -o config_template.json
    # (The current generate_config.py uses dummy data if no path specified)
    python generate_config.py <your_target_column_name> -o config_template.json
    ```
    This creates `config_template.json`. **You MUST review and edit this template.**

*   **Manual Editing:**
    1.  Rename `config_template.json` (or your generated file) to `config.json`.
    2.  Open `config.json` and edit the following fields:
        *   `"model_type"`: Set to `"classifier"` or `"regressor"`.
        *   `"model_path"`: **Crucial:** Set the relative path to your saved LightGBM model file (e.g., `"models/my_model.txt"`). The model should be saved using `booster.save_model()`.
        *   `"target_name"`: Name of the variable being predicted.
        *   `"target_type"`: `"categorical"` or `"numerical"`.
        *   `"class_names"`: **Required for Classifiers** if you want readable class names instead of indices. This must be a list of strings corresponding to the **exact order** of classes used during model training (often determined by `LabelEncoder`'s `.classes_` attribute or the order of categories if using Pandas `Categorical` type directly). Example: `["setosa", "versicolor", "virginica"]`.
        *   `"features"`: Review the list of features. Ensure `type` (`"numerical"` or `"categorical"`), `range` (for numerical hints), `options` (for categorical dropdowns), and `dtype` are correct based on your training data.

*   **Example `config.json` structure:**
    ```json
    {
      "model_type": "classifier",
      "model_path": "models/iris_lgbm_classifier.txt",
      "target_name": "species",
      "target_type": "categorical",
      "class_names": ["setosa", "versicolor", "virginica"],
      "features": [
        {
          "name": "sepal length (cm)",
          "type": "numerical",
          "range": [4.3, 7.9],
          "dtype": "float64"
        },
        {
          "name": "garden_location",
          "type": "categorical",
          "options": ["Mixed", "Shady", "Sunny"], // Order might matter depending on encoding
          "dtype": "category"
        }
        // ... other features
      ]
    }
    ```

## Training Your Model

1.  **Prepare Data:** Load your data into a Pandas DataFrame. Ensure categorical features are appropriately typed (e.g., `category` or `object`) and the target variable is correct.
2.  **Train Model:** Use LightGBM (`LGBMClassifier` or `LGBMRegressor`). Remember to handle categorical features correctly during training (e.g., using the `categorical_feature` parameter or `dtype='category'`).
3.  **Save Model:** Save the **booster** object, not the scikit-learn wrapper, using `model.booster_.save_model('path/to/your/model.txt')`. Place the saved model file (e.g., `my_model.txt`) inside the `models/` directory (or update `model_path` in `config.json` accordingly).
4.  **Generate/Update Config:** Use `generate_config.py` on your training data to get a template. **Crucially, update the `model_path` and `class_names` (for classifiers) in the resulting `config.json` file.** Verify all feature details.

    *See `train_example.py` for a practical demonstration.*

## Running the Application

1.  Make sure your `config.json` is correctly configured and points to a valid model file within the `models` directory.
2.  From the root directory of the project (where `app.py` is located), run:
    ```bash
    flask run
    ```
    Or, for more control (e.g., setting host/port):
    ```bash
    python app.py
    ```
    (The `app.py` script currently runs on `0.0.0.0:5000` with debug mode enabled).

3.  Open your web browser and navigate to `http://127.0.0.1:5000/` (or the appropriate host/port if you changed it).

## Using the Application

1.  The web page will display a form with input fields for each feature defined in `config.json`.
2.  For **numerical** features, enter a number. Hints about the typical range might be displayed.
3.  For **categorical** features, select an option from the dropdown menu.
4.  Click the **"Predict"** button.
5.  The application will process the input, run the prediction using the loaded LightGBM model, and display the result below the form.
6.  If there are input errors or prediction issues, an error message will be shown. Check the console logs where Flask is running for more details.

## Dependencies

*   Flask
*   LightGBM
*   Pandas
*   Numpy
*   Scikit-learn (primarily used in the example training script)

See `requirements.txt` for specific versions.

## Customization / Future Ideas

*   **Preprocessing:** Add more robust preprocessing steps within `app.py` if needed (e.g., scaling, encoding) based on how the model was trained. This might require storing preprocessing objects (like scalers or encoders) alongside the model.
*   **Model Types:** Extend to support other model libraries (e.g., Scikit-learn, XGBoost) by adding logic to `load_model` and potentially adjusting the prediction call.
*   **Input Methods:** Allow file uploads (e.g., CSV) for batch predictions.
*   **Error Handling:** Improve user feedback on errors.
*   **Styling:** Enhance the web interface appearance.
*   **Dockerization:** Package the application into a Docker container for easier deployment.