# ensemble_model.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from lightgbm import LGBMClassifier

def get_stacking_model(X_train_res, y_train_res, classification_type='binary'):
    if classification_type == 'binary':
        final_estimator = RandomForestClassifier(random_state=42)
    else:
        final_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)

    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgboost_model = xgb.XGBClassifier(random_state=42, objective='multi:softmax' if classification_type != 'binary' else 'binary:logistic')
    lgbm = LGBMClassifier(random_state=42)

    # Stacking Classifier
    estimators = [('rf', rf), ('gb', gb), ('xgboost', xgboost_model), ('lgbm', lgbm)]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

    # Hyperparameter Tuning (example for RandomForest)
    param_grid = {'rf__n_estimators': [100, 200],
                  'rf__max_depth': [10, 20, None],
                  'rf__min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_res, y_train_res)

    return grid_search.best_estimator_
