import matplotlib.pyplot as plt
import seaborn as sns
import logging


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import dice_ml
import logging
from typing import List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import logging
class LGBMWrapper:
    """
    A wrapper for a LightGBM model to ensure that the data types of the input
    for prediction match the original training data types. This is crucial for
    libraries like DiCE that may manipulate the data.
    """
    def __init__(self, model: lgb.LGBMClassifier, train_data: pd.DataFrame):
        """
        Initializes the LGBMWrapper.

        Args:
            model (lgb.LGBMClassifier): The trained LightGBM model.
            train_data (pd.DataFrame): The training data (features only) used to
                                       train the model. This is used to store
                                       the original data types.
        """
        self.model = model
        self.original_dtypes = train_data.dtypes
        self.feature_names = train_data.columns.tolist()

    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Converts the input data to a pandas DataFrame with the correct column
        names and data types.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input data for prediction.

        Returns:
            pd.DataFrame: The data converted to a DataFrame with original dtypes.
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()
        
        # Ensure all required columns are present
        for col in self.feature_names:
            if col not in X_df.columns:
                raise ValueError(f"Missing column in input data: {col}")

        return X_df.astype(self.original_dtypes)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts class probabilities for the input data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input data.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        X_converted = self._prepare_data(X)
        return self.model.predict_proba(X_converted)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts the class for the input data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input data.

        Returns:
            np.ndarray: The predicted classes.
        """
        X_converted = self._prepare_data(X)
        return self.model.predict(X_converted)

class CounterFactualExplainer:
    """
    A class to generate counterfactual explanations for a machine learning model's
    predictions using the DiCE library.

    It is currently optimized and tested for use with LightGBM classifiers.

    Usage:
        explainer = CounterFactualExplainer(model, train_data, target_col)
        df_dice_x = CounterFactualExplainer.obtain_data_to_explain(df)
        explainer.explain(df_dice_x.iloc[[0]])


    """
    def __init__(self, model: lgb.LGBMClassifier, train_data: pd.DataFrame, target_col: str):
        """
        Initializes the CounterFactualExplainer.

        Args:
            model (lgb.LGBMClassifier): The trained machine learning model.
                                       Currently, only LGBMClassifier is supported.
            train_data (pd.DataFrame): The dataset used for training the model.
                                       It should include the target column.
            target_col (str): The name of the target variable column.
        """
        import dice_ml
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must be a pandas DataFrame.")
        if target_col not in train_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in train_data.")

        self.model = model
        self.train_data = train_data.copy()
        self.target_col = target_col

        logging.info("Initializing CounterFactualExplainer.")
        
        # Prepare data for DiCE
        continuous_features = self.train_data.select_dtypes(include=np.number).columns.tolist()
        if self.target_col in continuous_features:
            continuous_features.remove(self.target_col)
        
        # Impute missing values
        for col in self.train_data.columns:
            if col in continuous_features:
                median_val = self.train_data[col].median()
                self.train_data[col] = self.train_data[col].fillna(median_val)
            else:
                mode_val = self.train_data[col].mode()
                if not mode_val.empty:
                    self.train_data[col] = self.train_data[col].fillna(mode_val[0])

        # Wrap the model for DiCE
        wrapped_model = LGBMWrapper(self.model, self.train_data.drop(columns=[self.target_col]))

        # Initialize DiCE
        d = dice_ml.Data(dataframe=self.train_data, continuous_features=continuous_features, outcome_name=self.target_col)
        m = dice_ml.Model(model=wrapped_model, backend="sklearn")
        
        # Using the 'random' method for generating counterfactuals
        self.dice = dice_ml.Dice(d, m, method='random')
        logging.info("CounterFactualExplainer initialized successfully.")

    def explain(self, instance: pd.DataFrame, desired_class: Union[str, int] = 'opposite', 
                total_cfs: int = 3, features_to_vary: Optional[List[str]] = None) -> None:
        """
        Generates and displays counterfactual explanations for a given instance.

        Args:
            instance (pd.DataFrame): A single row DataFrame of the instance to be explained.
            desired_class (Union[str, int], optional): The desired outcome for the
                                                       counterfactuals. Defaults to 'opposite'.
            total_cfs (int, optional): The number of counterfactuals to generate. Defaults to 3.
            features_to_vary (Optional[List[str]], optional): List of features that can be
                                                              changed to generate counterfactuals.
                                                              Defaults to None (all features).
        """
        if not isinstance(instance, pd.DataFrame) or not instance.shape[0] == 1:
            raise ValueError("The instance to explain must be a single-row pandas DataFrame.")

        logging.info(f"Generating {total_cfs} counterfactuals for the instance.")
        try:
            if features_to_vary is None:
                features_to_vary = self.train_data.columns.tolist()
                if self.target_col in features_to_vary:
                    features_to_vary.remove(self.target_col)
            elif not isinstance(features_to_vary, list):
                counterfactuals = self.dice.generate_counterfactuals(
                    instance,
                    total_CFs=total_cfs,
                    desired_class=desired_class,
                    features_to_vary=features_to_vary
                )
            
            print("\n" + "="*50)
            print("Counterfactual Explanation")
            print("="*50)
            
            print("\nOriginal Instance:")
            print(instance.to_string())

            if counterfactuals.cf_examples_list:
                print("\nCounterfactuals (showing only changes):")
                counterfactuals.visualize_as_dataframe(show_only_changes=True)
            else:
                print("\nNo counterfactuals found for the given constraints.")

            print("-" * 50 + "\n")

        except Exception as e:
            logging.error(f"Error during counterfactual explanation: {e}", exc_info=True)
            print(f"An error occurred while generating counterfactuals: {e}")
            print("-" * 50)

    @staticmethod
    def obtain_data_to_explain(model: lgb.LGBMClassifier, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        A helper function to extract instances from a dataframe that are
        predicted as the negative class (0).

        Args:
            model (lgb.LGBMClassifier): The trained model.
            df (pd.DataFrame): The dataframe to search for instances.
            target_col (str): The name of the target column.

        Returns:
            pd.DataFrame: A dataframe of instances predicted as class 0.
        """
        df_dice = df.copy()
        
        # Separate features and target
        X = df_dice.drop(columns=[target_col])
        y = df_dice[target_col]
        
        # Filter for instances that are actually class 0 and predicted as class 0
        is_class_0 = (y == 0)
        predicted_as_0 = (model.predict(X) == 0)
        
        df_to_explain = X[is_class_0 & predicted_as_0]
        
        return df_to_explain


def explain_with_model_property(model,feature_names,plot_top_n=10):
    """
    Explain the model using feature importance.
    This function is useful for tree-based models like RandomForest, XGBoost, etc.
    It calculates the feature importance and plots the top N features.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=feature_names)
    feature_importance = feature_importance.sort_values(ascending=False)
    print("Feature Importance:")
    print(feature_importance)
    



def shap_explain_model_on_batch(model, train_x, feature_names,excluded_cat_features=[]):
    """
    Explain the model using SHAP values.
    This function is useful for tree-based models like LightGBM, XGBoost, etc.
    """
    import shap
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_x)

    # Summary plot
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, train_x, feature_names=feature_names)
    plt.show()


    
    # 3. Identify categorical features
    X = train_x

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in excluded_cat_features]
    
    # 4. Create dependence plots for each categorical feature
    for feature in categorical_features:
        # 5. Create a violin plot for categorical features
        plt.figure(figsize=(14, 8))
        
        # Get unique categories
        categories = X[feature].unique()
        
        # Prepare data for violin plot
        cat_shap_values = []
        cat_names = []
        
        for cat in categories:
            indices = X[feature] == cat
            if np.any(indices):
                values = shap_values[indices, X.columns.get_loc(feature)]
                cat_shap_values.append(values)
                cat_names.append(str(cat))
        
        # Plot violin plot
        #plt.violinplot(cat_shap_values, showmeans=True, showmedians=True)
        plot_colored_violin(cat_shap_values)
        plt.xticks(range(1, len(cat_names) + 1), cat_names, rotation=45)
        plt.ylabel("SHAP Value")
        plt.title(f"Distribution of SHAP Values for each category in {feature}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    return shap_values
    

def shap_explain_model_on_single(model, sample, feature_names):
    """
    Explain the model using SHAP values for a single sample.
    This function is useful for tree-based models like LightGBM, XGBoost, etc.
    """
    import shap

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, feature_names=feature_names)



def plot_colored_violin(dataset, title="Violin Plot", palette="viridis", alpha=0.7):
    """
    Creates a violin plot with customizable colors for the violin bodies.

    Args:
        dataset: The data to plot. This can be a sequence of N arrays, or an (M,N) array,
                 or a single array. plt.violinplot will create N violins.
        title (str): The title of the plot.
        palette (str or list): Seaborn color palette name or a list of colors.
        alpha (float): Transparency of the violin bodies.
    """
    plt.figure(figsize=(10, 6))
    vp = plt.violinplot(dataset, showmeans=True, showmedians=True)

    # Determine the number of violins created
    num_violins = 0
    if isinstance(dataset, list):
        num_violins = len(dataset)
    elif isinstance(dataset, np.ndarray):
        if dataset.ndim == 1:
            num_violins = 1
        elif dataset.ndim == 2:
            num_violins = dataset.shape[1]
    
    if num_violins == 0 and vp['bodies']: # Fallback if dataset type is unusual but bodies exist
        num_violins = len(vp['bodies'])

    if num_violins > 0:
        colors = sns.color_palette(palette, num_violins)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i % len(colors)]) # Use modulo for safety
            body.set_edgecolor('black') # Or another contrasting color, or 'none'
            body.set_alpha(alpha)
    else: # Single violin or unable to determine multiple, color first body if it exists
        if vp['bodies']:
            vp['bodies'][0].set_facecolor(sns.color_palette(palette, 1)[0])
            vp['bodies'][0].set_edgecolor('black')
            vp['bodies'][0].set_alpha(alpha)


    # You can also customize the lines for means, medians, etc.
    if 'cmeans' in vp:
        vp['cmeans'].set_color('red')
        vp['cmeans'].set_linewidth(2)
        vp['cmeans'].set_linestyle('--')
    if 'cmedians' in vp:
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(2)
    if 'cbars' in vp: # For the whiskers
        vp['cbars'].set_edgecolor('grey')
    if 'cmins' in vp:
        vp['cmins'].set_edgecolor('grey')
    if 'cmaxes' in vp:
        vp['cmaxes'].set_edgecolor('grey')

    #plt.title(title, fontsize=16)
    # Add x-axis labels if you have multiple violins representing different categories
    if num_violins > 1 and isinstance(dataset, list): # Or if you have category names
        # Assuming you might have category names for each violin
        # For example: category_names = [f'Category {i+1}' for i in range(num_violins)]
        # plt.xticks(np.arange(1, num_violins + 1), category_names)
        pass # Add logic for xticks if needed based on how 'dataset' is structured

    #plt.ylabel("Value") # Or a more specific label
    #plt.grid(True, linestyle='--', alpha=0.7)
    #plt.tight_layout()
    #plt.show()

    return vp # Return the dictionary of artists if needed for further customization