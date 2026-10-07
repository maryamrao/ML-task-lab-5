"""Download the standard datasets used by this lab.
Run this once from the project root if the CSV/data files are missing.
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = {
    os.path.join(ROOT, 'Activity_1_Diabetes', 'diabetes.csv'):
        'https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv',
    os.path.join(ROOT, 'Graded_Task_1_Titanic', 'train.csv'):
        'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv',
    os.path.join(ROOT, 'Graded_Task_2_Car_Safety', 'car.data'):
        'https://raw.githubusercontent.com/vincm1/UCI_car_data/main/car.data',
}

for path, url in FILES.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print('Already exists:', path)
        continue
    print('Downloading:', url)
    urllib.request.urlretrieve(url, path)
    print('Saved:', path)
print('\nDataset setup complete.')
