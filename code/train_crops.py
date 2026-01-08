import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../csvs/preprocessed.csv')
x = df.drop(columns=['Unnamed: 0', 'label'])
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))
x_train, x_temp, y_train, y_temp = train_test_split(x, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
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
    n_jobs=-1
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
random_search.fit(x_train, y_train)
best_model = random_search.best_estimator_
print(random_search.best_params_)
# y_cv_pred = model.predict(x_cv)
y_cv_pred = best_model.predict(x_cv)
class_names = ['Rice', 'Cassava', 'Pineapple', 'Rubber', 'Oil palm',
               'Durian', 'Rambutan', 'Coconut', 'Mango', 'Longan',
               'Jackfruit', 'Mangosteen', 'Longkong', 'Reservoir', 'Others']
print(classification_report(y_cv, y_cv_pred, target_names=class_names))
cm = confusion_matrix(y_cv, y_cv_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
importances = best_model.feature_importances_
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