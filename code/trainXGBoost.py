import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('../csvs/cleaned.csv')
x = df.drop(columns=['Water'])
y = df['Water']
xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
xCV, xTest, yCV, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp)
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(xTrain, yTrain)
yCVPred = model.predict(xCV)
print(f'The precision score is {precision_score(yCV, yCVPred, average='binary'):.3f}.')
print(f'The recall score is {recall_score(yCV, yCVPred, average='binary'):.3f}.')
print(f'The F1 score is {f1_score(yCV, yCVPred, average='binary'):.3f}.')