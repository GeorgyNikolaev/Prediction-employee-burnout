import torch
from torch.utils.data import Dataset
import pandas as pd


def get_classes(df):
    classes = df.unique().tolist()
    classes = {i: class_name for i, class_name in enumerate(classes)}
    return classes


class HRAnalysisDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.employees_info = df
        self.employees_info['Attrition'] = self.employees_info['Attrition'].map({'Yes': 1, 'No': 0})
        travel_classes = get_classes(self.employees_info['BusinessTravel'])
        self.employees_info['BusinessTravel'] = self.employees_info['BusinessTravel'].map(travel_classes)
        department_classes = get_classes(self.employees_info['Department'])
        self.employees_info['Department'] = self.employees_info['Department'].map(department_classes)
        education_classes = get_classes(self.employees_info['EducationField'])
        self.employees_info['EducationField'] = self.employees_info['EducationField'].map(education_classes)
        self.employees_info["Gender"] = self.employees_info["Gender"].map({"Male": 1, "Female": 0})
        job_role_classes = get_classes(self.employees_info['JobRole'])
        self.employees_info['JobRole'] = self.employees_info['JobRole'].map(job_role_classes)
        marital_status_classes = get_classes(self.employees_info['MaritalStatus'])
        self.employees_info['MaritalStatus'] = self.employees_info['MaritalStatus'].map(marital_status_classes)
        self.employees_info["OverTime"] = self.employees_info["OverTime"].map({"Yes": 1, "No": 0})
        self.features = self.employees_info.drop(['Attrition', 'EmployeeNumber', "JobLevel", "YearsAtCompany", "StockOptionLevel"], axis=1).to_numpy()
        self.features = torch.tensor(self.features)
        self.target = torch.tensor(self.employees_info['Attrition'].to_numpy())

    def __len__(self):
        return len(self.employees_info)

    def __getitem__(self, idx):
        return (self.features[idx], self.target[idx])
