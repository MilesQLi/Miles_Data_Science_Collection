

from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import pandas as pd
from sklearn.utils import resample
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def cross_val_model_auto(df,target,exclude_cols=[]):
    x = df.drop(exclude_cols+[target],axis=1)
    y = df[target]
    cat_cols = x.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        x[col] = x[col].astype('category')
    model = LGBMClassifier(n_estimators=1000,verbosity= -1)
    scores = cross_val_score(model,x,y,cv=5)
    print("scores:",scores)
    return np.mean(scores), np.std(scores)


def stack_model_training(df, target_col,base_model_class = None, base_model_params = None, meta_model_class = None, meta_model_params = None,index_cols=[],cross_val=True):

    n_models = 10 
    base_models = []
    meta_data = []


    X = df.drop([target_col]+index_cols, axis=1)
    y = df[target_col]

    if base_model_class is None:
        base_model_class = XGBRegressor
        
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

    
    print("🚀 Training base models & collecting OOB predictions...")
    for i in tqdm(range(n_models)):
        # Bootstrap sample
        X_sample, y_sample = resample(X, y, replace=True, n_samples=len(X), random_state=i)
        selected_idx = set(X_sample.index)
        oob_idx = list(set(X.index) - selected_idx)
        if base_model_params is not None:
            model = base_model_class(**base_model_params)
        else:
            model = base_model_class()
    
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
    
    predicted_mean = meta_df.mean(axis=1)
    predicted_median = meta_df.median(axis=1)
    predicted_std = meta_df.std(axis=1)
    predicted_min = meta_df.min(axis=1)
    predicted_max = meta_df.max(axis=1)

    meta_df["predicted_mean"] = predicted_mean
    meta_df["predicted_std"] = predicted_std
    meta_df["predicted_min"] = predicted_min
    meta_df["predicted_max"] = predicted_max
    meta_df["predicted_median"] = predicted_median


    X_meta = X.loc[meta_df.index].copy()  
    y_meta = y.loc[meta_df.index]       
    
    # Combine original features + predictions from base models
    X_meta_final = pd.concat([X_meta, meta_df], axis=1).fillna(0)

    print("🎯 Training meta-model ...")
    if meta_model_class is None:
        meta_model_class = CatBoostRegressor
        meta_model_params = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "random_state": 42,
            "task_type": "GPU", 
            "verbose": 100
        }
    
    if meta_model_params is not None:
        meta_model = CatBoostRegressor(**meta_model_params)
    else:
        meta_model = CatBoostRegressor()
    meta_model.fit(X_meta_final, y_meta)

    if cross_val:
        print("🔬 Performing cross-validation on meta-model...")
        cv_results = cross_val_score(meta_model, X_meta_final, y_meta, cv=5)
        print(f"Cross-validation score of meta-model: {np.mean(cv_results)} ± {np.std(cv_results)}")

    return base_models, meta_model


def stack_model_predict(test_df,base_models,meta_model):
    print("📦 Predicting on test set...")
    X_test_meta = test_df.copy()
    
    for i, model in enumerate(base_models):
        X_test_meta[f"pred_model_{i}"] = model.predict(test_df)
    base_model_pred_cols = [X for X in X_test_meta.columns if 'pred_model_' in X]
    X_test_meta['predicted_mean'] = X_test_meta[base_model_pred_cols].mean(axis=1)
    X_test_meta['predicted_std'] = X_test_meta[base_model_pred_cols].std(axis=1)
    X_test_meta['predicted_median'] = X_test_meta[base_model_pred_cols].median(axis=1)
    X_test_meta['predicted_min'] = X_test_meta[base_model_pred_cols].min(axis=1)
    X_test_meta['predicted_max'] = X_test_meta[base_model_pred_cols].max(axis=1)
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
        model = LGBMClassifier(verbosity=-1, is_unbalance = unbalance)
    else:
        model = LGBMClassifier(**params, verbosity=-1, is_unbalance = unbalance)
    
    if valid_x is not None and valid_y is not None:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric="average_precision", callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds,first_metric_only =True)])
    else:
        model.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric="average_precision", callbacks=[lgb.log_evaluation(period=1), lgb.early_stopping(early_stopping_rounds,first_metric_only =True)])

    results = model.evals_result_
    # Extract the AUPRC scores for the validation set
    # The key 'valid_0' is the default name for the first eval_set
    if 'valid_0' in results and "average_precision" in results['valid_0']:
        using_validation = True
    else:
        using_validation = False
    if using_validation:
        validation_auprc = results['valid_0']["average_precision"]
    else:
        validation_auprc = results['training']["average_precision"]
    iterations = range(1, len(validation_auprc) + 1) # Iteration numbers (start from 1)

    # --- Step 5: Plot ---
    plt.figure(figsize=(10, 6))
    if using_validation:
        plt.plot(iterations, validation_auprc, label='Validation AUPRC', marker='.')
    else:
        plt.plot(iterations, validation_auprc, label='Training AUPRC', marker='.')
    plt.title('AUPRC vs. Boosting Iteration (Step)')
    plt.xlabel('Boosting Iteration (Step)')
    plt.ylabel('AUPRC')
    plt.xticks(np.arange(0, len(validation_auprc)+1, step=max(1, len(validation_auprc)//10))) # Adjust tick frequency
    plt.grid(True)
    plt.legend()
    plt.show()

    # Find the best iteration if early stopping was used
    if model.best_iteration_:
        print(f"\nBest Iteration (based on validation AUPRC): {model.best_iteration_}")
        print(f"Best Validation AUPRC: {model.best_score_['valid_0']['auc']:.4f}")
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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, average_precision_score
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
    average_precision = average_precision_score(test_y, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUPRC: {average_precision:.4f}")
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



def tune_lgbm_optuna(
    X_train, y_train,
    model_type='classifier',  # 'classifier' or 'regressor'
    n_trials=100,
    cv=5,  # Number of CV folds or a CV splitter object
    scoring=None, # Sklearn scoring string, e.g., 'roc_auc', 'neg_mean_squared_error'
    early_stopping_rounds=50, # For LGBM early stopping within trials
    fixed_params=None, # Dictionary of fixed parameters for LGBM
    X_eval=None, # Optional evaluation set for final model training early stopping
    y_eval=None, # Optional evaluation set for final model training early stopping
    random_state=42
):
    """
    Finetunes LGBMClassifier or LGBMRegressor hyperparameters using Optuna.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target.
        model_type (str): 'classifier' or 'regressor'.
        n_trials (int): Number of Optuna optimization trials.
        cv (int or CV splitter): Cross-validation strategy.
        scoring (str, optional): Scorer string. Defaults based on model_type.
                                 For classifier: 'roc_auc' (binary) or 'accuracy' (multiclass).
                                 For regressor: 'neg_mean_squared_error'.
        early_stopping_rounds (int, optional): LGBM early stopping rounds. If None, disabled.
        fixed_params (dict, optional): Parameters to fix for LGBM model.
        X_eval (pd.DataFrame or np.ndarray, optional): Evaluation features for final model's early stopping.
        y_eval (pd.Series or np.ndarray, optional): Evaluation target for final model's early stopping.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (best_params, best_model)
               best_params (dict): Dictionary of the best hyperparameters.
               best_model (lgb.LGBMClassifier or lgb.LGBMRegressor): Trained model with best params.
    """
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, mean_squared_error, r2_score
    import numpy as np
    import pandas as pd # For isinstance checks if X, y are pandas Series/DataFrame

    if fixed_params is None:
        fixed_params = {}

    # Determine default objective, metric, and Optuna direction based on model_type
    if model_type == 'classifier':
        # Check if it's binary or multiclass classification
        if isinstance(y_train, (pd.Series, np.ndarray)):
            num_classes = len(np.unique(y_train))
        else: # Assume list-like
            num_classes = len(set(y_train))

        if num_classes == 2:
            default_objective = 'binary'
            default_lgbm_metric = 'binary_logloss' # or 'auc'
            if scoring is None:
                scoring = 'roc_auc'
            optuna_direction = 'maximize'
        else:
            default_objective = 'multiclass'
            default_lgbm_metric = 'multi_logloss' # or 'multi_error'
            fixed_params['num_class'] = num_classes # Required for multiclass
            if scoring is None:
                scoring = 'accuracy' # Or f1_macro, etc.
            # For metrics like accuracy, f1_score, roc_auc_ovr -> maximize
            # For metrics like log_loss -> minimize
            if scoring in ['accuracy', 'f1_weighted', 'f1_macro', 'f1_micro', 'roc_auc_ovr', 'roc_auc_ovo']:
                optuna_direction = 'maximize'
            elif scoring in ['neg_log_loss']: # sklearn's log_loss is negative
                 optuna_direction = 'maximize'
            else: # e.g. if user passes 'log_loss' which is not sklearn standard for cross_val_score
                print(f"Warning: scoring '{scoring}' for multiclass. Assuming higher is better. Adjust optuna_direction if needed.")
                optuna_direction = 'maximize'

    elif model_type == 'regressor':
        default_objective = 'regression'
        default_lgbm_metric = 'rmse' # or 'l2', 'mae'
        if scoring is None:
            scoring = 'neg_mean_squared_error'
        # For neg_mean_squared_error, neg_root_mean_squared_error, r2 -> maximize
        # For mean_squared_error, root_mean_squared_error, mean_absolute_error -> minimize
        if scoring in ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']:
            optuna_direction = 'maximize'
        elif scoring in ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error']:
            optuna_direction = 'minimize'
            print(f"Warning: Scoring '{scoring}' implies minimization. Make sure Optuna direction is set correctly if you override this logic.")
        else:
            print(f"Warning: scoring '{scoring}' for regression. Assuming higher is better. Adjust optuna_direction if needed.")
            optuna_direction = 'maximize'
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    # --- Objective Function for Optuna ---
    def objective(trial):
        # Define search space for hyperparameters
        param_grid = {
            "objective": default_objective,
            "metric": default_lgbm_metric, # Metric for LGBM internal eval, can differ from `scoring`
            "random_state": random_state,
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05), # Bagging fraction
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0, step=0.05), # Feature fraction
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), # L1
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True), # L2
            # "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]), # Can add if desired
        }

        if model_type == 'classifier' and num_classes == 2: # Only for binary
            param_grid["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        
        # Incorporate fixed parameters
        param_grid.update(fixed_params)

        # --- Model Instantiation ---
        if model_type == 'classifier':
            model = lgb.LGBMClassifier(**param_grid)
        else: # regressor
            model = lgb.LGBMRegressor(**param_grid)

        # --- Cross-validation with Early Stopping and Pruning ---
        fit_params = {}
        callbacks = []

        if early_stopping_rounds:
            # Note: cross_val_score doesn't easily support dynamic eval_sets per fold for early stopping.
            # For robust early stopping with pruning, a manual CV loop is better.
            # However, for simplicity here, we'll use LightGBMPruningCallback which works with
            # LGBM's internal CV or if eval_set is passed to fit.
            # For cross_val_score, early stopping is applied *after* the split, using a portion of the
            # training data of that fold as eval set, which is not ideal.
            # A more advanced setup would loop through CV folds manually.

            # Let's create a proper CV splitter
            if isinstance(cv, int):
                if model_type == 'classifier':
                    # Stratified K-Folds for classification
                    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    # Regular K-Folds for regression
                    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            else: # User provided a CV splitter object
                cv_splitter = cv

            scores = []
            # Manual CV loop to properly handle early stopping and pruning with distinct eval sets
            for train_idx, val_idx in cv_splitter.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx], \
                                           X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx], \
                                           y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                
                current_callbacks = []
                if early_stopping_rounds:
                    current_callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=-1))
                    # Pruning callback for Optuna
                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l2" if default_lgbm_metric == "rmse" else default_lgbm_metric) # Use lgbm's metric
                    current_callbacks.append(pruning_callback)

                model.fit(X_fold_train, y_fold_train,
                          eval_set=[(X_fold_val, y_fold_val)],
                          eval_metric=default_lgbm_metric, # Use lgbm's metric name
                          callbacks=current_callbacks)

                # Make predictions and score
                if scoring == 'roc_auc' and model_type == 'classifier': # Requires predict_proba
                    preds = model.predict_proba(X_fold_val)[:, 1]
                    score = roc_auc_score(y_fold_val, preds)
                elif scoring == 'accuracy' and model_type == 'classifier':
                    preds = model.predict(X_fold_val)
                    score = accuracy_score(y_fold_val, preds)
                elif scoring == 'neg_mean_squared_error' and model_type == 'regressor':
                    preds = model.predict(X_fold_val)
                    score = -mean_squared_error(y_fold_val, preds) # Optuna maximizes
                elif scoring == 'r2' and model_type == 'regressor':
                    preds = model.predict(X_fold_val)
                    score = r2_score(y_fold_val, preds)
                else: # Fallback to generic scorer if not explicitly handled
                    # This requires a custom scorer or ensuring 'scoring' matches sklearn's names
                    # For simplicity, we'll assume the common ones are handled.
                    # If you use a different scorer, you might need to adjust this part.
                    # Example: y_pred = model.predict(X_fold_val)
                    # scikit_scorer = get_scorer(scoring)
                    # score = scikit_scorer(model, X_fold_val, y_fold_val) # This would refit, not ideal
                    # Better to predict then score:
                    if hasattr(model, "predict_proba"): # Classifier
                        try:
                            preds_proba = model.predict_proba(X_fold_val)
                            # Handle multiclass for scorers like roc_auc_ovr
                            if scoring.startswith('roc_auc_') and preds_proba.shape[1] > 2:
                                score = make_scorer(globals()[scoring.split("_")[0] + "_score"], needs_proba=True, **({'average': scoring.split("_")[-1]} if len(scoring.split("_")) > 2 else {}))(model, X_fold_val, y_fold_val)
                            else:
                                score = make_scorer(globals()[scoring.replace('neg_','').split("_")[0] + "_score"], needs_proba=True)(model, X_fold_val, y_fold_val)

                        except: # Fallback to predict
                            preds = model.predict(X_fold_val)
                            score_func_name = scoring.replace('neg_','') # e.g. neg_mean_squared_error -> mean_squared_error
                            score_val = globals()[score_func_name + "_score"](y_fold_val, preds)
                            if scoring.startswith('neg_'):
                                score = -score_val
                            else:
                                score = score_val
                    else: # Regressor
                        preds = model.predict(X_fold_val)
                        score_func_name = scoring.replace('neg_','')
                        score_val = globals()[score_func_name + "_score"](y_fold_val, preds)
                        if scoring.startswith('neg_'):
                            score = -score_val
                        else:
                            score = score_val
                scores.append(score)
            
            # If all folds were pruned, this might be empty or contain NaNs
            if not scores or np.isnan(np.mean(scores)):
                return -np.inf if optuna_direction == 'maximize' else np.inf # Penalize heavily

            return np.mean(scores)

        else: # No early stopping, use simple cross_val_score
            # Note: Pruning is not effective without early stopping in this setup
            if isinstance(cv, int):
                if model_type == 'classifier':
                    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                cv_splitter = cv
            
            score = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring=scoring, fit_params=None)
            return score.mean()

    # --- Run Optuna Study ---
    study = optuna.create_study(direction=optuna_direction, pruner=optuna.pruners.MedianPruner() if early_stopping_rounds else None)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest trial score ({scoring}): {study.best_value}")
    print(f"Best hyperparameters: {best_params}")

    # --- Train Final Model with Best Parameters ---
    final_params = {
        "objective": default_objective,
        "metric": default_lgbm_metric,
        "random_state": random_state,
    }
    final_params.update(fixed_params) # Add user-fixed params
    final_params.update(best_params)  # Add Optuna-found params (will override defaults if overlapping)
    
    if 'num_class' in fixed_params: # Ensure num_class from fixed_params is used if multiclass
        final_params['num_class'] = fixed_params['num_class']


    if model_type == 'classifier':
        best_model = lgb.LGBMClassifier(**final_params)
    else:
        best_model = lgb.LGBMRegressor(**final_params)

    fit_final_params = {}
    final_callbacks = []
    if early_stopping_rounds and X_eval is not None and y_eval is not None:
        print("Training final model with early stopping using provided X_eval, y_eval.")
        fit_final_params['eval_set'] = [(X_eval, y_eval)]
        fit_final_params['eval_metric'] = default_lgbm_metric
        final_callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False if n_trials > 1 else True)) # Verbose only if not many trials
        fit_final_params['callbacks'] = final_callbacks
    elif early_stopping_rounds:
        print("Warning: `early_stopping_rounds` is set, but no `X_eval`, `y_eval` provided for final model training.")
        print("Final model will be trained without early stopping on the full dataset, using n_estimators from best trial.")
        # n_estimators is already in final_params from best_params

    best_model.fit(X_train, y_train, **fit_final_params)
    
    if early_stopping_rounds and best_model.best_iteration_:
        print(f"Final model trained with {best_model.best_iteration_} iterations due to early stopping.")
        # Update n_estimators in best_params to reflect the actual number used
        best_params['n_estimators'] = best_model.best_iteration_


    return best_params, best_model
