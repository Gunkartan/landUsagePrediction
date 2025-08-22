import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

df = pd.read_csv("allYears.csv")
cols_to_use = df.columns[:9].tolist() + ['Class']
df = df[cols_to_use]
df = df.dropna()
print(df.shape)

for col in df.select_dtypes(include=['object', 'category']):
    if col != 'Class':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

le_target = LabelEncoder()
y = le_target.fit_transform(df['Class'])
X = df.drop(columns=['Class'])
X_train_cv, X_test, y_train_cv, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_cv, y_train, y_cv = train_test_split(
    X_train_cv, y_train_cv, test_size=0.25, random_state=42, stratify=y_train_cv
)
dtrain = xgb.DMatrix(X_train, label=y_train)
dcv = xgb.DMatrix(X_cv, label=y_cv)
dtest = xgb.DMatrix(X_test, label=y_test)
num_classes = len(np.unique(y))
params = {
    "objective": "multi:softmax",
    "num_class": num_classes,
    "max_depth": 5,
    "eta": 0.1,
    "eval_metric": "mlogloss"
}
evals = [(dtrain, "train"), (dcv, "cv")]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=True
)
y_pred = bst.predict(dtest)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")