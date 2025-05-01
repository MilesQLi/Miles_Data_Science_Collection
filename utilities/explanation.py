


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

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_x)

    # Summary plot
    shap.summary_plot(shap_values, train_x, feature_names=feature_names)
    

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