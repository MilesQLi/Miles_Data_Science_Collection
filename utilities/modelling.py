

def training_binary_classification_with_lgbm(train_x,train_y,valid_x=None,valid_y=None,params=None,split=None,early_stopping_rounds=100):
    """
    Training a binary classification model using LightGBM.
    """
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

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
        model = LGBMClassifier(verbose=0, is_unbalance = unbalance)
    else:
        model = LGBMClassifier(**params, verbose=0, is_unbalance = unbalance)
    
    if valid_x is not None and valid_y is not None:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[lgb.early_stopping(early_stopping_rounds)])
    else:
        model.fit(train_x, train_y, verbose=0)
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