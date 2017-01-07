import pandas
from sklearn.ensemble import RandomForestClassifier

df = pandas.read_csv('train.csv')

df['Gender'] = df.Sex.map({'male': 1, 'female': 0}).astype(int)


df['AgeFill'] = df['Age']

for gender_bin in range(2):
    for pclass in range(1, 4):
        df.loc[(df.Age.isnull()) & (df.Gender == gender_bin) & (df.Pclass == pclass), 'AgeFill'] = df[(df.Gender == gender_bin) & (df.Pclass == pclass)].Age.dropna().median()


df['AgeIsNull'] = pandas.isnull(df.Age).astype(int)
df['FamilySize'] = df.SibSp + df.Parch
df['Age*Class'] = df.AgeFill * df.Pclass
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId', 'AgeIsNull'], axis=1)
train_data = df.values

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
importances = forest.feature_importances_
print importances