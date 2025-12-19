import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

df = pd.read_csv('../csvs/cleanedCropWithMTCI.csv')
x = df.drop(columns=['Label', 'Crops'])
y = df['Crops']
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
classes = np.unique(y_train)
raw_weights = compute_class_weight('balanced', classes=classes, y=y_train)
scaled_weights = np.log1p(raw_weights)
scaled_weights = scaled_weights / scaled_weights.min()
weight_map = dict(zip(classes, scaled_weights))
sample_weights = np.array([weight_map[c] for c in y_train])
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
    n_jobs=-1
)
model.fit(x_train, y_train, sample_weight=sample_weights)
# search.fit(x_train, y_train)
# print(search.best_params_)
# print(search.best_score_)
# best_model = search.best_estimator_
# y_cv_pred = model.predict(x_cv)
proba = model.predict_proba(x_cv)
y_true = y_cv.values
thresholds = np.linspace(0.2, 0.9, 15)
best_t = []

for c in range(14):
    best_f, best_thr = 0, 0

    for t in thresholds:
        preds = np.full_like(y_true, 13)
        preds[proba[:, c] > t] = c
        f = f1_score(y_true, preds, average='macro')

        if f > best_f:
            best_f, best_thr = f, t

    best_t.append(best_thr)
    # print(f'{c} | {best_thr} | {best_f:.3f}')

best_preds = np.full_like(y_true, 13)
passed = np.zeros_like(proba, dtype=bool)

for c, t in enumerate(best_t):
    passed[:, c] = proba[:, c] > t

for i in range(len(y_true)):
    valid_classes = np.where(passed[i])[0]

    if len(valid_classes) > 0:
        best_class = valid_classes[np.argmax(proba[i, valid_classes])]
        best_preds[i] = best_class

# y_cv_pred = best_model.predict(x_cv)
class_names = ['Rice', 'Cassava', 'Pineapple', 'Rubber', 'Oil palm',
               'Durian', 'Rambutan', 'Coconut', 'Mango', 'Longan',
               'Jackfruit', 'Mangosteen', 'Longkong', 'Others']
print(classification_report(y_cv, best_preds, target_names=class_names))
xgb.plot_importance(model, importance_type='gain')
plt.title('Feature importance.')
plt.tight_layout()
plt.show()
cm = confusion_matrix(y_cv, best_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix.')
plt.xlabel('Predicted labels.')
plt.ylabel('True labels.')
plt.tight_layout()
plt.show()