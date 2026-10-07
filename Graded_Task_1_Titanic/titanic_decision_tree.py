import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Graded_Task_1_Titanic', 'train.csv')
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(DATA):
    raise FileNotFoundError('train.csv not found. Run download_datasets.py first.')

print('\n' + '='*60)
print('GRADED TASK 1 - TITANIC SURVIVAL')
print('='*60)

df = pd.read_csv(DATA)
# Normalize possible capitalization from common Titanic copies.
rename = {c.lower(): c for c in df.columns}
if 'survived' not in df.columns and 'survived' in rename:
    df = df.rename(columns={rename['survived']: 'Survived'})
for wanted in ['Pclass','Sex','Age','SibSp','Parch','Fare']:
    if wanted not in df.columns:
        match = next((c for c in df.columns if c.lower() == wanted.lower()), None)
        if match: df = df.rename(columns={match:wanted})

print('\nFirst 5 rows:')
print(df.head())
print('\nShape:', df.shape)
print('\nMissing values:')
print(df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']].isnull().sum())

features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
work = df[features + ['Survived']].copy()
work['Age'] = work['Age'].fillna(work['Age'].median())
work['Sex'] = work['Sex'].map({'male': 0, 'female': 1})
work = work.dropna()

X = work[features]
y = work['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('\nBasic Decision Tree Accuracy:', accuracy_score(y_test, pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, pred))
print('\nClassification Report:\n', classification_report(y_test, pred, zero_division=0))

plt.figure(figsize=(22, 12))
plot_tree(model, feature_names=features, class_names=['Did not survive','Survived'], filled=True, rounded=True, fontsize=7)
plt.title('Titanic - Basic Decision Tree')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'titanic_decision_tree.png'), dpi=160)
plt.close()

pruned = DecisionTreeClassifier(max_depth=4, random_state=1)
pruned.fit(X_train, y_train)
pred2 = pruned.predict(X_test)
print('\nPruned Decision Tree Accuracy:', accuracy_score(y_test, pred2))

plt.figure(figsize=(18, 10))
plot_tree(pruned, feature_names=features, class_names=['Did not survive','Survived'], filled=True, rounded=True, fontsize=9)
plt.title('Titanic - Pruned Decision Tree (max_depth=4)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'titanic_pruned_tree.png'), dpi=180)
plt.close()

print('\nExample predictions (first 10 test passengers):')
for i, (actual, predicted) in enumerate(zip(y_test.iloc[:10], pred2[:10]), 1):
    print(f'Passenger {i}: actual={int(actual)}, predicted={int(predicted)}')
print('\nSaved Titanic tree images in outputs/.')
