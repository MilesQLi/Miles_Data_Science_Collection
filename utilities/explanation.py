


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
    



def shap_explain_model_on_batch(model, train_x, feature_names):
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
                values = shap_values.values[indices, X.columns.get_loc(feature)]
                cat_shap_values.append(values)
                cat_names.append(str(cat))
        
        # Plot violin plot
        plt.violinplot(cat_shap_values, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(cat_names) + 1), cat_names, rotation=45)
        plt.ylabel("SHAP Value")
        plt.title(f"Distribution of SHAP Values for each category in {feature}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    

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