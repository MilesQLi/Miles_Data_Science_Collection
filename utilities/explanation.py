import matplotlib.pyplot as plt
import seaborn as sns


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