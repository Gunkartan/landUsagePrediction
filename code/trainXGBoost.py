import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('../csvs/cleaned.csv')
x = df.drop(columns=['Water'])
y = df['Water']
xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp)
model = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(xTrain, yTrain, eval_set=[(xVal, yVal)], verbose=True)
yPred = model.predict(xVal)
print(f'The precision score is {precision_score(yVal, yPred):.4f}.')
print(f'The recall score is {recall_score(yVal, yPred):.4f}.')
print(f'The F1 score is {f1_score(yVal, yPred):.4f}.')