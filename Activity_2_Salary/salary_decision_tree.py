import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Activity_2_Salary', 'salaries.csv')
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

print('\n' + '='*60)
print('ACTIVITY 2 - SALARY PREDICTION')
print('='*60)

df = pd.read_csv(DATA)
print('\nDataset:')
print(df)

inputs = df.drop('salary_more_then_100k', axis='columns').copy()
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

print('\nLabel mappings:')
print('Company:', dict(zip(le_company.classes_, le_company.transform(le_company.classes_))))
print('Job:', dict(zip(le_job.classes_, le_job.transform(le_job.classes_))))
print('Degree:', dict(zip(le_degree.classes_, le_degree.transform(le_degree.classes_))))

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
print('\nEncoded inputs:')
print(inputs_n)

model = DecisionTreeClassifier(random_state=1)
model.fit(inputs_n, target)
print('\nModel Score:', model.score(inputs_n, target))

# Use the encoder rather than hard-coded numeric labels.
def predict_salary(company, job, degree):
    row = [[le_company.transform([company])[0], le_job.transform([job])[0], le_degree.transform([degree])[0]]]
    return int(model.predict(row)[0])

for degree in ['bachelors', 'masters']:
    result = predict_salary('google', 'computer programmer', degree)
    print(f'\nGoogle + Computer Programmer + {degree.title()}')
    print('Salary > 100K:', 'Yes' if result == 1 else 'No')

plt.figure(figsize=(16, 9))
plot_tree(model, feature_names=['company_n','job_n','degree_n'], class_names=['No','Yes'], filled=True, rounded=True, fontsize=9)
plt.title('Salary Prediction Decision Tree')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'salary_decision_tree.png'), dpi=180)
plt.close()
print('\nSaved salary_decision_tree.png in outputs/.')
