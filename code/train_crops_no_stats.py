import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score

df = pd.read_csv('../csvs/cleanedCropAllFeatures.csv')
x = df.drop(columns=['Label', 'Crops'])
y = df['Crops']
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    n_jobs=-1,
    num_class=14,
    objective='multi:softprob',
    random_state=42
)
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=15,
    scoring='f1_macro',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=1
)
model.fit(x_train, y_train)
# search.fit(x_train, y_train)
# print(search.best_params_)
# print(search.best_score_)
# best_model = search.best_estimator_
# y_cv_pred = model.predict(x_cv)
proba = model.predict_proba(x_cv)
best_score = 0

for t in np.linspace(0.1, 0.9, 9):
    crop_mask = np.max(proba[:, :-1], axis=1) < t
    preds = np.argmax(proba, axis=1)
    preds[crop_mask] = 13
    f = f1_score(y_cv, preds, average='macro')

    if f > best_score:
        best_t = t
        best_score = f
        best_preds = preds

print(f'{t} | {f:.3f}')
# y_cv_pred = best_model.predict(x_cv)
class_names = ['Rice', 'Cassava', 'Pineapple', 'Rubber', 'Oil palm',
               'Durian', 'Rambutan', 'Coconut', 'Mango', 'Longan',
               'Jackfruit', 'Mangosteen', 'Longkong', 'Others']
print(classification_report(y_cv, best_preds, target_names=class_names))