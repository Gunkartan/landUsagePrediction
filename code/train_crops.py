import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv('../csvs/preprocessed.csv')
x = df.drop(columns=['Unnamed: 0', 'label'])
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))
x_train, x_temp, y_train, y_temp = train_test_split(x, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
sample_weights = np.array([class_weight_dict[y] for y in y_train])
param_dist = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    n_estimators=800,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=0,
    reg_lambda=1
)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=15,
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
# model.fit(x_train, y_train)
model.fit(x_train, y_train, sample_weight=sample_weights)
# random_search.fit(x_train, y_train)
# best_model = random_search.best_estimator_
# print(random_search.best_params_)
y_cv_pred = model.predict(x_cv)
# y_cv_pred = best_model.predict(x_cv)
# y_cv_proba = model.predict_proba(x_cv)
# class_thresholds = np.zeros(num_classes)
# threshold_candidates = np.arange(0.1, 0.91, 0.05)

# for cls in range(num_classes):
#     best_f = 0
#     best_t = 0

#     for t in threshold_candidates:
#         y_pred = []

#         for i in range(len(y_cv_proba)):
#             p = y_cv_proba[i]

#             if p[cls] >= t:
#                 y_pred.append(cls)

#             else:
#                 y_pred.append(np.argmax(p))

#         y_pred = np.array(y_pred)
#         f = f1_score(y_cv, y_pred, average='weighted')

#         if f > best_f:
#             best_f = f
#             best_t = t

#     class_thresholds[cls] = best_t

class_names = ['Rice', 'Cassava', 'Pineapple', 'Rubber', 'Oil palm',
               'Durian', 'Rambutan', 'Coconut', 'Mango', 'Longan',
               'Jackfruit', 'Mangosteen', 'Longkong', 'Reservoir', 'Others']

# for i, t in enumerate(class_thresholds):
#     print(f'{class_names[i]}, {t:.2f}.')

for k, v in class_weight_dict.items():
    print(f'{class_names[k]}, {v:.3f}.')

# y_cv_pred_tuned = []

# for p in y_cv_proba:
#     assigned = False

#     for cls in range(num_classes):
#         if p[cls] >= class_thresholds[cls]:
#             y_cv_pred_tuned.append(cls)
#             assigned = True

#             break

#     if not assigned:
#         y_cv_pred_tuned.append(np.argmax(p))

# y_cv_pred_tuned = np.array(y_cv_pred_tuned)
print(classification_report(y_cv, y_cv_pred, target_names=class_names))
# print(classification_report(y_cv, y_cv_pred_tuned, target_names=class_names))
print(f1_score(y_cv, y_cv_pred, average='weighted'))
# print(f1_score(y_cv, y_cv_pred_tuned, average='weighted'))
cm = confusion_matrix(y_cv, y_cv_pred)
# cm = confusion_matrix(y_cv, y_cv_pred_tuned)
cm_normalized = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
importances = model.feature_importances_
# importances = best_model.feature_importances_
feature_names = (x.columns if isinstance(x, pd.DataFrame) else [f'Feature {i}' for i in range(x.shape[1])])
fi = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
fi.head(15)
plt.figure(figsize=(8, 6))
sns.barplot(data=fi.head(15), x='Importance', y='Feature')
plt.title('Feature importance')
plt.show()