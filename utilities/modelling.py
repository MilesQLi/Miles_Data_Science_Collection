

from sklearn.model_selection import KFold
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import pandas as pd
from sklearn.utils import resample
from xgboost import XGBRegressor
from catboost import CatBoostRegressor



def stack_model_training(df,target_col,index_cols=[]):

    n_models = 10 
    base_models = []
    meta_data = []


    X = df.drop([target_col]+index_cols, axis=1)
    y = df[target_col]
    
    print("🚀 Training base models & collecting OOB predictions...")
    for i in tqdm(range(n_models)):
        # Bootstrap sample
        X_sample, y_sample = resample(X, y, replace=True, n_samples=len(X), random_state=i)
        selected_idx = set(X_sample.index)
        oob_idx = list(set(X.index) - selected_idx)
        base_model_params = {
        'max_depth': 10,
        'n_estimators': 2000,
        'learning_rate': 0.0049,
        'subsample': 0.91,
        'colsample_bytree': 0.86,
        'gamma': 0.0014,
        'reg_alpha': 0.025,
        'reg_lambda': 0.0106,
        'min_child_weight': 7,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': 'cuda'
        }
        model = XGBRegressor(**base_model_params)
    
        model = model.fit(X_sample, y_sample)
    
        base_models.append(model)
    
        # Predict on OOB samples  ->for meta model  data  
        if oob_idx:
            X_oob = X.loc[oob_idx]
            y_oob = y.loc[oob_idx]
            preds_oob = model.predict(X_oob)
    
            meta_data.append(pd.DataFrame({
                "id": X_oob.index,
                f"pred_model_{i}": preds_oob
            }).set_index("id"))
    
    # Merge all OOB predictions by outer join (union)  
    print("🔗 Merging OOB predictions...")
    meta_df = pd.concat(meta_data, axis=1)
    meta_df = meta_df.groupby(meta_df.index).first() 
    X_meta = X.loc[meta_df.index].copy()  
    y_meta = y.loc[meta_df.index]       
    
    # Combine original features + predictions from base models
    X_meta_final = pd.concat([X_meta, meta_df], axis=1).fillna(0)

    print("🎯 Training meta-model (CatBoost)...")
    meta_model_params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_state": 42,
        "task_type": "GPU", 
        "verbose": 100
    }
    meta_model = CatBoostRegressor(**meta_model_params)
    meta_model.fit(X_meta_final, y_meta)
    return base_models, meta_model


def stack_model_predict(test_df,base_models,meta_model):
    print("📦 Predicting on test set...")
    X_test_meta = test_df.copy()
    
    for i, model in enumerate(base_models):
        X_test_meta[f"pred_model_{i}"] = model.predict(test_df)
    
    final_preds = meta_model.predict(X_test_meta)
    return final_preds






def OOF_predictions(model_class, train_x, train_y, test_x, params=None, early_stopping_rounds = 100, n_splits=5):
    """
    Out-of-Fold (OOF) predictions using KFold cross-validation.
    """


    oof_train = np.zeros(train_x.shape[0])
    oof_test = np.zeros(test_x.shape[0])
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_index, valid_index in kf.split(train_x):
        model = model_class(**params) if params else model_class()
        X_train, X_valid = train_x.iloc[train_index], train_x.iloc[valid_index]
        y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='l2', callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds)])
        
        oof_train[valid_index] = model.predict(X_valid)
        oof_test += model.predict(test_x) / n_splits
    
    print(f"OOF Train RMSE: {np.sqrt(np.mean((oof_train - train_y) ** 2)):.4f}")

    return oof_train, oof_test


def training_regression_with_lgbm(train_x,train_y,valid_x=None,valid_y=None,params=None,split=None,early_stopping_rounds=100):
    """
    Training a regression model using LightGBM.
    """
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

    if valid_x is None and split is not None:
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split, random_state=42)
    
    object_cols = train_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        train_x[col] = train_x[col].astype('category')
        if valid_x is not None:
            valid_x[col] = valid_x[col].astype('category')

    if params is None:
        model = LGBMRegressor(verbose=0)
    else:
        model = LGBMRegressor(**params, verbose=0)
    
    if valid_x is not None and valid_y is not None:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='l2', callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds)])
    else:
        model.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='l2', callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds)])

    results = model.evals_result_
    if 'valid_0' in results and 'l2' in results['valid_0']:
        using_validation = True
    else:
        using_validation = False
    
    if using_validation:
        validation_l2 = results['valid_0']['l2']
    else:
        validation_l2 = results['training']['l2']
    iterations = range(1, len(validation_l2) + 1) # Iteration numbers (start from 1)
    plt.figure(figsize=(10, 6))
    if using_validation:
        plt.plot(iterations, validation_l2, label='Validation L2', marker='.')
    else:
        plt.plot(iterations, validation_l2, label='Training L2', marker='.')
    plt.title('L2 vs. Boosting Iteration (Step)')
    plt.xlabel('Boosting Iteration (Step)')
    plt.ylabel('L2 Loss')

    return model


def training_binary_classification_with_lgbm(train_x,train_y,valid_x=None,valid_y=None,params=None,split=None,early_stopping_rounds=100):
    """
    Training a binary classification model using LightGBM.
    """
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np

    val_count = train_y.value_counts(normalize=True)
    if val_count[0] > 0.6:
        unbalance = True
    else:
        unbalance = False   
    
    print(f"Class unbalanced: {unbalance}")  
    if valid_x is None and split is not None:
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split, random_state=42, stratify=train_y)
    
    object_cols = train_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        train_x[col] = train_x[col].astype('category')
        if valid_x is not None:
            valid_x[col] = valid_x[col].astype('category')

    if params is None:
        model = LGBMClassifier(verbose=5, is_unbalance = unbalance)
    else:
        model = LGBMClassifier(**params, verbose=5, is_unbalance = unbalance)
    
    if valid_x is not None and valid_y is not None:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='auc', callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds,first_metric_only =True)])
    else:
        model.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds,first_metric_only =True)])

    results = model.evals_result_
    # Extract the AUROC scores for the validation set
    # The key 'valid_0' is the default name for the first eval_set
    if 'valid_0' in results and 'auc' in results['valid_0']:
        using_validation = True
    else:
        using_validation = False
    if using_validation:
        validation_auroc = results['valid_0']['auc']
    else:
        validation_auroc = results['training']['auc']
    iterations = range(1, len(validation_auroc) + 1) # Iteration numbers (start from 1)

    # --- Step 5: Plot ---
    plt.figure(figsize=(10, 6))
    if using_validation:
        plt.plot(iterations, validation_auroc, label='Validation AUROC', marker='.')
    else:
        plt.plot(iterations, validation_auroc, label='Training AUROC', marker='.')
    plt.title('AUROC vs. Boosting Iteration (Step)')
    plt.xlabel('Boosting Iteration (Step)')
    plt.ylabel('AUROC')
    plt.xticks(np.arange(0, len(validation_auroc)+1, step=max(1, len(validation_auroc)//10))) # Adjust tick frequency
    plt.grid(True)
    plt.legend()
    plt.show()

    # Find the best iteration if early stopping was used
    if model.best_iteration_:
        print(f"\nBest Iteration (based on validation AUROC): {model.best_iteration_}")
        print(f"Best Validation AUROC: {model.best_score_['valid_0']['auc']:.4f}")
    return model

def training_multi_classification_with_lgbm(train_x,train_y,valid_x=None,valid_y=None,params=None,split=None,early_stopping_rounds=100):
    """
    Training a multi-class classification model using LightGBM.
    """
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    if valid_x is None and split is not None:
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split, random_state=42, stratify=train_y)
    
    object_cols = train_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        train_x[col] = train_x[col].astype('category')
        if valid_x is not None:
            valid_x[col] = valid_x[col].astype('category')

    if params is None:
        model = LGBMClassifier(verbose=0, class_weight = 'balanced')
    else:
        model = LGBMClassifier(**params, verbose=0, class_weight = 'balanced')
    
    if valid_x is not None and valid_y is not None:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[lgb.early_stopping(early_stopping_rounds)])
    else:
        model.fit(train_x, train_y, verbose=0)
    return model

def training_regression_with_lgbm(train_x,train_y,valid_x=None,valid_y=None,params=None,split=None,early_stopping_rounds=100):
    """
    Training a regression model using LightGBM.
    """
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

    if valid_x is None and split is not None:
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split, random_state=42)
    
    object_cols = train_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        train_x[col] = train_x[col].astype('category')
        if valid_x is not None:
            valid_x[col] = valid_x[col].astype('category')

    if params is None:
        model = LGBMRegressor(verbose=0)
    else:
        model = LGBMRegressor(**params, verbose=0)
    
    if valid_x is not None and valid_y is not None:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[lgb.early_stopping(early_stopping_rounds)])
    else:
        model.fit(train_x, train_y, verbose=0)
    return model


def evaluate_binary_classification_model(model, name, test_x, test_y):

    """
    Evaluate a binary classification model using various metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
    from matplotlib import pyplot as plt
    import seaborn as sns

    object_cols = test_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        test_x[col] = test_x[col].astype('category')
    print(f"\n--- Evaluating {name} ---")
    y_pred = model.predict(test_x)
    y_pred_proba = model.predict_proba(test_x)[:, 1] # Probability of class 1

    # Calculate Metrics
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    roc_auc = roc_auc_score(test_y, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(test_y, y_pred))
    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Conversion', 'Conversion'], yticklabels=['No Conversion', 'Conversion'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_y, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


def evaluate_multi_classification_model(model, name, test_x, test_y):
    """
    Evaluate a multi-class classification model using various metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sns

    object_cols = test_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        test_x[col] = test_x[col].astype('category')
    print(f"\n--- Evaluating {name} ---")
    y_pred = model.predict(test_x)

    # Calculate Metrics
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='weighted')
    recall = recall_score(test_y, y_pred, average='weighted')
    f1 = f1_score(test_y, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(test_y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

def evaluate_regression_model(model, name, test_x, test_y):
    """
    Evaluate a regression model using various metrics.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from matplotlib import pyplot as plt
    import seaborn as sns

    object_cols = test_x.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        test_x[col] = test_x[col].astype('category')
    print(f"\n--- Evaluating {name} ---")
    y_pred = model.predict(test_x)

    # Calculate Metrics
    mse = mean_squared_error(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Residual Plot
    sns.residplot(x=y_pred, y=test_y - y_pred, lowess=True)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{name} - Residual Plot')
    plt.show()