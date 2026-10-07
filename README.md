# Decision Tree Machine Learning Lab

**Student:** Maryam Ishfaq  
**Roll No:** FA23-BSE-037

This project contains the two solved lab activities and two graded lab tasks based on Decision Tree Classification.

## Included

1. Activity 1 — Diabetes Prediction
2. Activity 2 — Salary Prediction
3. Graded Task 1 — Titanic Survival Prediction
4. Graded Task 2 — Car Safety Prediction

## Folder Structure

```text
Decision_Tree_Lab/
├── Activity_1_Diabetes/
│   ├── diabetes_decision_tree.py
│   └── diabetes.csv (downloaded by setup script)
├── Activity_2_Salary/
│   ├── salary_decision_tree.py
│   └── salaries.csv
├── Graded_Task_1_Titanic/
│   ├── titanic_decision_tree.py
│   └── train.csv (downloaded by setup script)
├── Graded_Task_2_Car_Safety/
│   ├── car_safety_decision_tree.py
│   └── car.data (downloaded by setup script)
├── outputs/
├── download_datasets.py
├── requirements.txt
└── README.md
```

## Installation

Open PowerShell/Terminal in this project folder:

```bash
pip install -r requirements.txt
```

Then download the datasets:

```bash
python download_datasets.py
```

## Run Each Program

```bash
python Activity_1_Diabetes/diabetes_decision_tree.py
python Activity_2_Salary/salary_decision_tree.py
python Graded_Task_1_Titanic/titanic_decision_tree.py
python Graded_Task_2_Car_Safety/car_safety_decision_tree.py
```

## Outputs

The programs automatically create PNG decision-tree visualizations in `outputs/`.

## Main Concepts

- Decision Tree Classification
- Training and testing data
- Accuracy
- Confusion Matrix
- Classification Report
- Label Encoding
- Entropy
- Gini impurity
- `max_depth` and pruning

## Dataset Sources

- Pima Indians Diabetes dataset: standard Pima dataset mirror.
- Titanic: Kaggle Titanic training dataset mirror.
- Car Evaluation: UCI Car Evaluation dataset mirror.
- Salary: the 16-row salary dataset used in the supplied lab activity.

## Submission Screenshot Checklist

- [ ] Activity 1 code/output and tree image
- [ ] Activity 2 code/output and prediction results
- [ ] Titanic accuracy, confusion matrix/classification report and tree
- [ ] Car accuracy, confusion matrix/classification report and tree
- [ ] README and project structure
