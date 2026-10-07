import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Activity_1_Diabetes', 'diabetes.csv')
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(DATA):
    raise FileNotFoundError('diabetes.csv not found. Run download_datasets.py first.')

print('\n' + '='*60)
print('ACTIVITY 1 - DIABETES PREDICTION')
print('='*60)

df = pd.read_csv(DATA)
# Support both the standard named Pima CSV and the lab's headerless format.
expected = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
if 'Outcome' in df.columns:
    df = df.rename(columns={'Pregnancies':'pregnant','Glucose':'glucose','BloodPressure':'bp',
                            'SkinThickness':'skin','Insulin':'insulin','BMI':'bmi',
                            'DiabetesPedigreeFunction':'pedigree','Age':'age','Outcome':'label'})
elif not set(expected).issubset(df.columns):
    df = pd.read_csv(DATA, header=None, names=expected)

print('\nFirst 5 rows:')
print(df.head())
print('\nDataset shape:', df.shape)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = df[feature_cols]
y = pd.to_numeric(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

basic = DecisionTreeClassifier(random_state=1)
basic.fit(X_train, y_train)
pred = basic.predict(X_test)
print('\nBasic Decision Tree Accuracy:', accuracy_score(y_test, pred))

plt.figure(figsize=(22, 12))
plot_tree(basic, feature_names=feature_cols, class_names=['0','1'], filled=True, rounded=True, fontsize=7)
plt.title('Diabetes - Basic Decision Tree')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diabetes_basic_tree.png'), dpi=160)
plt.close()

pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
pruned.fit(X_train, y_train)
pred2 = pruned.predict(X_test)
acc2 = accuracy_score(y_test, pred2)
print('Pruned Decision Tree Accuracy:', acc2)
print('\nConfusion Matrix:\n', confusion_matrix(y_test, pred2))
print('\nClassification Report:\n', classification_report(y_test, pred2, zero_division=0))

plt.figure(figsize=(18, 10))
plot_tree(pruned, feature_names=feature_cols, class_names=['0','1'], filled=True, rounded=True, fontsize=9)
plt.title('Diabetes - Pruned Decision Tree (Entropy, max_depth=3)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diabetes_pruned_tree.png'), dpi=180)
plt.close()
print('Saved tree images in outputs/.')
