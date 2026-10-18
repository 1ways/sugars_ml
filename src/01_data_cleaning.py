import pandas as pd
import numpy as np

col_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv('data/pima-indians-diabetes.csv', names=col_names)

duplicates_count = df.duplicated().sum()
print(f"Duplicates count: {duplicates_count}")
df = df.drop_duplicates()

df['PatientID'] = range(1, len(df) + 1)

print("\n--- Data types ---")
print(df.dtypes)

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

Q1 = df[cols_with_zeros].quantile(0.25)
Q3 = df[cols_with_zeros].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[cols_with_zeros] < (Q1 - 1.5 * IQR)) | (df[cols_with_zeros] > (Q3 + 1.5 * IQR))).sum()
print(outliers)

for col in cols_with_zeros:
    df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 150], labels=['Young', 'Adult', 'Senior'])

monitoring_lookup = pd.DataFrame({
    'AgeGroup': ['Young', 'Adult', 'Senior'],
    'CheckupFrequency': ['Every 2 years', 'Yearly', 'Every 6 months']
})

df_clean = pd.merge(df, monitoring_lookup, on='AgeGroup', how='left')

df_clean.to_csv('data/diabetes_clean.csv', index=False)
print("\nThe cleaned data has been successfully saved to 'data/diabetes_clean.csv'")