import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv('../csvs/cleanedCrop.csv')
x = df.drop(columns=['Label', 'Crops'])
y, uniques = pd.factorize(df['Crops'])
print(dict(enumerate(uniques)))
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
num_classes = len(np.unique(y))
print(f'The detected number of classes is {num_classes}.')
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.2,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob',
    num_class=num_classes
)
random = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=100,
    scoring='f1_weighted',
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
y_cv_pred = model.predict(x_cv)
class_names = ['Rice', 'Cassava', 'Pineapple', 'Rubber', 'Oil palm',
               'Durian', 'Rambutan', 'Coconut', 'Mango', 'Longan',
               'Jackfruit', 'Mangosteen', 'Longkong', 'Others']
print(classification_report(y_cv, y_cv_pred, target_names=class_names))
# print(confusion_matrix(y_cv, y_cv_pred))
# plt.figure(figsize=(10, 8))
# plot_importance(model, importance_type='gain')
# plt.title('Feature importance.')
# plt.tight_layout()
# plt.show()