import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

df = pd.read_csv('../csvs/cleaned.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
xCV, xTest, yCV, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp)
print(f'{len(xTrain)} | {len(xCV)} | {len(xTest)}')
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(xTrain, yTrain)
yCVPred = model.predict(xCV)
precision = precision_score(yCV, yCVPred)
recall = recall_score(yCV, yCVPred)
f1 = f1_score(yCV, yCVPred)
print(f'{precision:.3f} | {recall:.3f} | {f1:.3f}')
print(classification_report(yCV, yCVPred))