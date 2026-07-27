"""
Generates a synthetic salary dataset and saves it as data/salary_data.csv

Features:
- YearsExperience : Number of years of work experience
- EducationLevel  : Bachelors / Masters / PhD
- Age             : Age of the employee

Target:
- Salary
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 500  # number of records

years_experience = np.round(np.random.uniform(0, 25, N), 1)
age = np.round(years_experience + np.random.uniform(21, 26, N))
education_level = np.random.choice(
    ["Bachelors", "Masters", "PhD"], size=N, p=[0.55, 0.35, 0.10]
)

education_bonus = {"Bachelors": 0, "Masters": 15000, "PhD": 30000}
edu_bonus_arr = np.array([education_bonus[e] for e in education_level])

# Base formula + noise to make it realistic
base_salary = 30000
salary = (
    base_salary
    + years_experience * 3500
    + edu_bonus_arr
    + np.random.normal(0, 5000, N)  # noise
)
salary = np.round(np.clip(salary, 25000, None), 2)

df = pd.DataFrame({
    "YearsExperience": years_experience,
    "EducationLevel": education_level,
    "Age": age,
    "Salary": salary
})

os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "salary_data.csv")
df.to_csv(out_path, index=False)
print(f"Dataset saved to {out_path}")
print(df.head())
