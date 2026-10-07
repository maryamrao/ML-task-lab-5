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

## Screenshots
<img width="1127" height="671" alt="image" src="https://github.com/user-attachments/assets/d9cbc71b-d428-48cb-8e3c-5dd1a82d73d0" />
<img width="1224" height="743" alt="image" src="https://github.com/user-attachments/assets/ab970128-99cf-4f15-bff5-5950d11e3c5f" />
<img width="1124" height="729" alt="image" src="https://github.com/user-attachments/assets/182d0400-0e1f-4dfd-a016-b28634103cac" />
<img width="1247" height="668" alt="image" src="https://github.com/user-attachments/assets/5b3d4bf4-d073-47d0-abdb-43c8f970245c" />
<img width="1242" height="617" alt="image" src="https://github.com/user-attachments/assets/a6faa20d-293a-4b44-a5dc-94950564af45" />
<img width="1158" height="753" alt="image" src="https://github.com/user-attachments/assets/37657a26-9813-4530-942f-fe1043b2264a" />
<img width="1107" height="712" alt="image" src="https://github.com/user-attachments/assets/8613cc0f-03df-42d6-aa26-5643715b10e3" />
<img width="1218" height="659" alt="image" src="https://github.com/user-attachments/assets/00041b09-378b-48da-9f5d-ccf450a415b3" />
<img width="1260" height="708" alt="image" src="https://github.com/user-attachments/assets/ecc5c3e5-fdb3-42cf-b0ab-285db0367735" />

