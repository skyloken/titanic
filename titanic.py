import pandas as pd
import numpy as np

train = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
# %%
train.head()
# %%
test.head()

# %%
train_shape = train.shape
test_shape = test.shape
print(train_shape)
print(test_shape)

# %%
train.describe()
# %%
test.describe()

# %%
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns

# %%
kesson_table(train)

# %%
kesson_table(test)

# %%
# カテゴリデータの処理
import numpy as np
sex_mapping = {'male': 0, 'female': 1}
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train['Sex'] = train['Sex'].map(sex_mapping)
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

# %%
# 欠損データの補完
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values="NaN", strategy="median", axis=0)
train[['Age']] = imr.fit_transform(train[['Age']])
test[['Age']] = imr.fit_transform(test[['Age']])
test[['Fare']] = imr.fit_transform(test[['Fare']])
imr = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
train[['Embarked']] = imr.fit_transform(train[['Embarked']])
test[['Embarked']] = imr.fit_transform(test[['Embarked']])

# %%
kesson_table(train)
# %%
kesson_table(test)

# %%
# 決定木
from sklearn.tree import DecisionTreeClassifier
X_train = train[['Pclass', 'Sex', 'Age', 'Fare']].values
y_train = train['Survived'].values
# 決定木の作成
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
# testの説明変数の値を取得
X_test = test[['Pclass', 'Sex', 'Age', 'Fare']].values
# testの説明変数を使って予測
pred = tree.predict(X_test)
# PassengerIdを取得
PassengerId = np.array(test['PassengerId'].astype(int))
my_solution = pd.DataFrame(pred, PassengerId, columns=['Survived'])
# CSV保存
my_solution.to_csv('outputs/tree.csv', index_label=['PassengerId'])

# %%
# 特徴量を増やす
X_train2 = train[['Pclass', 'Age', 'Sex',
                  'Fare', 'SibSp', 'Parch', 'Embarked']]
tree2 = DecisionTreeClassifier(
    max_depth=10, min_samples_split=5, random_state=1)
tree2.fit(X_train2, y_train)
X_test2 = test[['Pclass', 'Age', 'Sex', 'Fare',
                'SibSp', 'Parch', 'Embarked']].values
pred = tree2.predict(X_test2)
my_solution = pd.DataFrame(pred, PassengerId, columns=['Survived'])
my_solution.to_csv('outputs/tree2.csv', index_label=['PassengerId'])

# %%
#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=1, n_jobs=-1)
forest.fit(X_train2, y_train)
pred = forest.predict(X_test2)
my_solution = pd.DataFrame(pred, PassengerId, columns=['Survived'])
my_solution.to_csv('outputs/forest.csv', index_label=['PassengerId'])
