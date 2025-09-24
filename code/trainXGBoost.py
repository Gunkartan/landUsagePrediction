import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from collections import Counter

df = pd.read_csv('../csvs/cleaned.csv')
x = df.drop(columns=['Label', 'Water'])
y = df['Water']
xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
xCV, xTest, yCV, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp)
# params = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }
counter = Counter(yTrain)
neg, pos = counter[0], counter[1]
scale = neg / pos
print(f'The positive weight will be scaled by {scale}.')
bestParams = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 1.0
}
model = xgb.XGBClassifier(
    **bestParams,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=3, #Using 3 yields the best result so far.
    use_label_encoder=False,
    random_state=42
)
# random = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=params,
#     n_iter=30,
#     scoring='f1',
#     cv=3,
#     verbose=1,
#     n_jobs=1,
#     random_state=42
# )
model.fit(xTrain, yTrain)
# print(f'The best set of hyperparameters is {random.best_params_}.')
# print(f'The best CV F1 score is {random.best_score_:.3f}.')
# best = random.best_estimator_
# yCVPred = model.predict(xCV)
yCVProb = model.predict_proba(xCV)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(yCV, yCVProb)
ap = average_precision_score(yCV, yCVProb)
thresholdList = [0.5, 0.55, 0.6, 0.65]

for threshold in thresholdList:
    preds = (yCVProb >= threshold).astype(int)
    print(f'The threshold is {threshold}. The scores are below.')
    print(f'The precision score is {precision_score(yCV, preds, average='binary'):.3f}.')
    print(f'The recall score is {recall_score(yCV, preds, average='binary'):.3f}.')
    print(f'The F1 score is {f1_score(yCV, preds, average='binary'):.3f}.')

# fn = (yCV == 1) & (yCVPred == 0)
# missedClasses = df.loc[yCV.index[fn], 'Label']
# missedCounts = missedClasses.value_counts()
# print(f'The precision score is {precision_score(yCV, yCVPred, average='binary'):.3f}.')
# print(f'The recall score is {recall_score(yCV, yCVPred, average='binary'):.3f}.')
# print(f'The F1 score is {f1_score(yCV, yCVPred, average='binary'):.3f}.')
# print(f'The number of false negatives by class is below.')
# print(missedCounts)
plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, label=f'The AP is {ap:.3f}.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-recall curve.')
plt.legend()
plt.grid(True)
plt.show()