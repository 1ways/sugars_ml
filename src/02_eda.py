import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

output_dir = '../report/figures'
os.makedirs(output_dir, exist_ok=True)

data_path = '../data/diabetes_clean.csv'
if not os.path.exists(data_path):
    data_path = 'data/diabetes_clean.csv'

df = pd.read_csv(data_path)
print("Dataset Shape")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

print("First 5 Rows")
print(df.head(), "\n")

features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

print("1. Target Class Distribution:")
outcome_counts = df['Outcome'].value_counts()
outcome_pct = df['Outcome'].value_counts(normalize=True) * 100
balance_df = pd.DataFrame({
    'Count': outcome_counts, 
    'Percentage (%)': outcome_pct.round(2)
})
balance_df.index = ['0 (No Diabetes)', '1 (Diabetes)']
print(balance_df, "\n")

if 'BMICategory' in df.columns:
    print("BMI Category Distribution:")
    bmi_counts = df['BMICategory'].value_counts()
    bmi_pct = df['BMICategory'].value_counts(normalize=True) * 100
    bmi_df = pd.DataFrame({
        'Count': bmi_counts,
        'Percentage (%)': bmi_pct.round(2)
    })
    print(bmi_df, "\n")

plt.figure(figsize=(7, 5))
ax = sns.countplot(
    data=df, x='Outcome', hue='Outcome', 
    palette=['#4C72B0', '#DD8452'], legend=False
)
plt.title('Class Distribution of Outcome', fontsize=14, fontweight='bold')
plt.xlabel('Outcome')
plt.ylabel('Number of Patients')
plt.xticks([0, 1], ['No Diabetes (0)', 'Diabetes (1)'])

for p in ax.patches:
    height = int(p.get_height())
    if height > 0:
        ax.annotate(
            f'{height} ({height / len(df) * 100:.1f}%)', 
            (p.get_x() + p.get_width() / 2., height), 
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
plt.ylim(0, 580)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_class_balance.png', dpi=300)
plt.close()

print("2. Descriptive Statistics By Outcome: ")
stats_summary = df.groupby('Outcome')[features].agg(['mean', 'median', 'std']).round(2)
print(stats_summary.T, "\n")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.histplot(
        data=df, x=col, hue='Outcome', kde=True, element="step", 
        palette=['#4C72B0', '#DD8452'], ax=axes[i]
    )
    axes[i].set_title(f'Distribution of {col}', fontweight='bold')

plt.suptitle('Feature Distributions by Diabetes Outcome', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_histograms_by_class.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(
        data=df, x='Outcome', y=col, hue='Outcome', 
        palette=['#4C72B0', '#DD8452'], ax=axes[i], legend=False
    )
    axes[i].set_title(f'Boxplot of {col}', fontweight='bold')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['No (0)', 'Yes (1)'])

plt.suptitle('Feature Boxplots by Diabetes Outcome', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_boxplots_by_class.png', dpi=300, bbox_inches='tight')
plt.close()

key_vars = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
pairplot_fig = sns.pairplot(
    df[key_vars], hue='Outcome', palette=['#4C72B0', '#DD8452'],
    diag_kind='kde', corner=True, plot_kws={'alpha': 0.6}
)
pairplot_fig.fig.suptitle('Pairplot of Key Clinical Features', fontsize=16, fontweight='bold', y=1.02)
pairplot_fig.savefig(f'{output_dir}/04_pairplot_key_features.png', dpi=300, bbox_inches='tight')
plt.close()

print("3. Pearson Correlation Matrix")
corr_matrix = df[features + ['Outcome']].corr()
print(corr_matrix.round(3), "\n")

print("4. Feature Correlation With Outcome")
outcome_corr = corr_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)
print(outcome_corr.round(3), "\n")

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.2, vmax=1.0, 
    square=True, linewidths=0.5, cbar_kws={"shrink": .8}
)
plt.title('Pearson Correlation Heatmap of Medical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_correlation_heatmap.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=outcome_corr.values, y=outcome_corr.index, 
    hue=outcome_corr.index, palette='viridis', legend=False
)
plt.title('Feature Correlation with Diabetes', fontsize=14, fontweight='bold')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Features')
plt.xlim(0, 0.55)

for i, v in enumerate(outcome_corr.values):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/06_outcome_correlations_bar.png', dpi=300)
plt.close()

print(f"All 6 figures saved to '{output_dir}/'")