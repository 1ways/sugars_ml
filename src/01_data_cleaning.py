import pandas as pd
import numpy as np

col_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

df = pd.read_csv('data/pima-indians-diabetes.csv', names=col_names)

print(df.info())
print(df.describe())
print(df.head())

df['PatientID'] = range(1, len(df) + 1)

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

for col in cols_with_zeros:
    df[col] = df[col].fillna(df[col].median())

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

bmi_categories = pd.DataFrame({
    'PatientID': df['PatientID'],
    'BMICategory': df['BMI'].apply(categorize_bmi)
})

df_clean = pd.merge(df, bmi_categories, on='PatientID', how='left')

df_clean = df_clean.drop_duplicates()

df_clean.to_csv('data/diabetes_clean.csv', index=False)
print("The cleaned data has been successfully saved to 'diabetes_clean.csv'")