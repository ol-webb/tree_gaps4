import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import recall_score, make_scorer
import json

feature_directory = 'feature_datasets/'
timeseries_directory = 'local_saved_datasets/timeseries_datasets/'

dataframe_names = [
    'X_df_2yr', 'X_df_2yr_frost','X_df_2yr_lee',
    'X_df_2yr_temporal_frost','X_df_2yr_temporal',
    'X_df_2yr_temporal_lee','X_df_2yr_quegan',
    'X_df_2yr_quegan_frost','X_df_2yr_quegan3d'
]

feature_dataframes_list = []
for name in dataframe_names:
    filepath = os.path.join(feature_directory, f"{name}.csv")
    df = pd.read_csv(filepath)
    feature_dataframes_list.append(df)

param_grid = {
    'n_estimators': [200],  
    'max_depth': [3], 
    'learning_rate': [0.1, 0.15, 0.2], 
    'subsample': [0.75, 0.85, 0.9], 
    'colsample_bytree': [0.5, 0.6, 0.7], 
    'alpha': [0, 0.01, 0.1], 
    'gamma': [0.05, 0.1, 0.2],
    'lambda': [0, 1, 5], 
    'min_child_weight': [1, 2, 5], 
    'scale_pos_weight': [1.2, 1.5, 1.8],
}

recall_scorer = make_scorer(recall_score, average='macro')

best_recall = 0
best_model = None
best_params = None


for df in feature_dataframes_list:
    X = df.drop(columns=['gap','area'],axis=1)
    y = df['target'] 

    
    recall_scores = []
    for random_state in [42, 100, 200]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        model = xgb.XGBClassifier(objective='binary:logistic')

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=recall_scorer, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.best_estimator_.predict(X_test)
        recall = recall_score(y_test, y_pred, average='macro')
        recall_scores.append(recall)

    avg_recall = np.mean(recall_scores)

    # Save the best model parameters if current is better
    if avg_recall > best_recall:
        best_recall = avg_recall
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

# Save the best model parameters to a .json file
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)

# Save the best recall score to a separate .json file
with open('best_recall.json', 'w') as f:
    json.dump({"best_recall": best_recall}, f)