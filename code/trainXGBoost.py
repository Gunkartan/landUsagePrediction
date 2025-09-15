import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('../csvs/cleaned.csv')
x = df.drop(columns=['Water'])
y = df['Water']
xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
xCV, xTest, yCV, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp)
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=1
)
random = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=30,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=1,
    random_state=42
)
random.fit(xTrain, yTrain)
print(f'The best set of hyperparameters is {random.best_params_}.')
print(f'The best CV F1 score is {random.best_score_:.3f}.')
best = random.best_estimator_
yCVPred = best.predict(xCV)
print(f'The precision score is {precision_score(yCV, yCVPred, average='binary'):.3f}.')
print(f'The recall score is {recall_score(yCV, yCVPred, average='binary'):.3f}.')
print(f'The F1 score is {f1_score(yCV, yCVPred, average='binary'):.3f}.')