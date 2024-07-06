# PRODIGY_DS_3-Product-Purchase-Classifier

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
data = pd.read_csv(' https://archive.ics.uci.edu/ml/datasets/Bank+Marketing')
categorical_columns = data.select_dtypes(include=['object']).columns
def one_hot_encode(df, columns):
    for column in columns:
        one_hot = pd.get_dummies(df[column], prefix=column, drop_first=True)
        df = df.drop(column, axis=1)
        df = df.join(one_hot)
    return df
data = one_hot_encode(data, categorical_columns)
X = data.drop('y', axis=1)
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)


train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
