import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

select_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare', 'Cabin']

X_train = train[select_features]
X_test = test[select_features]

y_train = train['Survived']

# 因为embarked有缺失，我们选择较多的那个值填充，概率问题。(这个属性是登船的港口)
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

# Age采取取平均值(可以改进),fare(票价)
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# cabin
X_train['Cabin'].fillna('UNKNOWN', inplace=True)
X_test['Cabin'].fillna('UNKNOWN', inplace=True)

# 特征向量化
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 随机森林
rfc = RandomForestClassifier()
# XGBoost
xgbc = XGBClassifier()

# 交叉验证（5折）
print(cross_val_score(rfc, X_train, y_train, cv=5).mean())
print(cross_val_score(xgbc, X_train, y_train, cv=5).mean())

# 默认配置随机森林预测并存储结果
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('./submission_improve/rfc_submission.csv', index=False)

# 默认配置XGBC预测并存储结果
xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('./submission_improve/xgbc_submission.csv', index=False)

# 使用网格搜索,提高XGBC，存出结果
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1, refit=True)
gs.fit(X_train, y_train)
print(gs.best_score_)
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('./submission_improve/xgbc_best_submission.csv', index=False)
