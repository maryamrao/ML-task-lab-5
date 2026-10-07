import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Graded_Task_2_Car_Safety', 'car.data')
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(DATA):
    raise FileNotFoundError('car.data not found. Run download_datasets.py first.')

print('\n' + '='*60)
print('GRADED TASK 2 - CAR SAFETY')
print('='*60)

columns = ['buying','maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv(DATA, header=None, names=columns)
print('\nFirst 5 rows:')
print(df.head())
print('\nShape:', df.shape)
print('\nClass distribution:')
print(df['class'].value_counts())

X_text = df.drop('class', axis='columns').copy()
y = df['class'].copy()
encoders = {}
X = X_text.copy()
for col in X.columns:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col])
    print(f'{col} mapping:', dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_))))

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print('class mapping:', dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=1, stratify=y_encoded)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('\nBasic Decision Tree Accuracy:', accuracy_score(y_test, pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, pred))
print('\nClassification Report:\n', classification_report(y_test, pred, target_names=target_encoder.classes_, zero_division=0))

plt.figure(figsize=(24, 14))
plot_tree(model, feature_names=list(X.columns), class_names=target_encoder.classes_, filled=True, rounded=True, fontsize=6)
plt.title('Car Evaluation - Basic Decision Tree')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'car_decision_tree.png'), dpi=160)
plt.close()

pruned = DecisionTreeClassifier(max_depth=5, random_state=1)
pruned.fit(X_train, y_train)
pred2 = pruned.predict(X_test)
print('\nPruned Decision Tree Accuracy:', accuracy_score(y_test, pred2))

plt.figure(figsize=(20, 12))
plot_tree(pruned, feature_names=list(X.columns), class_names=target_encoder.classes_, filled=True, rounded=True, fontsize=7)
plt.title('Car Evaluation - Pruned Decision Tree (max_depth=5)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'car_pruned_tree.png'), dpi=180)
plt.close()

print('\nExample predictions:')
for row in X_test.iloc[:5].values:
    prediction = target_encoder.inverse_transform([pruned.predict([row])[0]])[0]
    print(dict(zip(columns[:-1], row)), '=>', prediction)
print('\nSaved Car Evaluation tree images in outputs/.')
