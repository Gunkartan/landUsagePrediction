import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('../csvs/cleanedTrimmed.csv')
x = df.drop(columns=['Label', 'Water'])
y = df['Water']
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
# pos = y_train.sum()
# neg = len(y_train) - pos
# scale = neg / pos
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.2,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=1.0,
    random_state=42,
    # use_label_encoder=False,
    # eval_metrics='logloss',
    scale_pos_weight=2, #Using 2 gives the best result so far.
    n_jobs=-1
)
random = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=100,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=1,
    random_state=42
)
model.fit(x_train, y_train)
# random.fit(x_train, y_train)
# print(f'The best set of hyperparameters is {random.best_params_}.')
# print(f'The best CV F1 score is {random.best_score_:.3f}.')
# best_model = random.best_estimator_
y_cv_prob = model.predict_proba(x_cv)[:, 1]
threshold_list = np.linspace(0, 1, 101)
results = []

for threshold in threshold_list:
    preds = (y_cv_prob >= threshold).astype(int)
    p = precision_score(y_cv, preds, zero_division=0)
    r = recall_score(y_cv, preds, zero_division=0)
    f = f1_score(y_cv, preds, zero_division=0)
    results.append([threshold, p, r, f])

metrics = pd.DataFrame(results, columns=['Thresholds', 'Precision', 'Recall', 'F1'])
valid = metrics[(metrics['Precision'] >= 0.8) | (metrics['Recall'] >= 0.8)]

if not valid.empty:
    best_score = valid.loc[valid['F1'].idxmax()]
    print(f'The best threshold is {best_score['Thresholds']:.3f}.')
    print(f'The precision is {best_score['Precision']:.3f}.')
    print(f'The recall is {best_score['Recall']:.3f}.')
    print(f'The F1 is {best_score['F1']:.3f}.')

else:
    print('There are no thresholds with a precision or recall of more than 0.8.')

# plt.figure(figsize=(10, 8))
# plot_importance(model, importance_type='gain')
# plt.title('Feature importance.')
# plt.tight_layout()
# plt.show()