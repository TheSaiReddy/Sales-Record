import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

sns.set()

# ===================== LOAD DATASET =====================
df = pd.read_csv("10000 Sales Records.csv")

# ===================== DATA PREPROCESSING =====================
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ===================== HISTOGRAM =====================
plt.figure()
plt.hist(df['Units Sold'], bins=30)
plt.title("Histogram of Units Sold")
plt.show()

# ===================== BAR CHART =====================
plt.figure()
df.groupby('Region')['Total Profit'].mean().plot(kind='bar')
plt.title("Average Profit by Region")
plt.show()

# ===================== LINE CHART =====================
plt.figure()
df_sorted = df.sort_values('Order Date')
plt.plot(df_sorted['Units Sold'])
plt.title("Units Sold Over Time")
plt.show()

# ===================== PIE CHART =====================
plt.figure()
df['Region'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Region Distribution")
plt.show()

# ===================== BOX PLOT =====================
plt.figure()
sns.boxplot(x=df['Total Profit'])
plt.title("Boxplot of Total Profit")
plt.show()

# ===================== SCATTER PLOT =====================
plt.figure()
plt.scatter(df['Units Sold'], df['Total Profit'])
plt.title("Units Sold vs Profit")
plt.xlabel("Units Sold")
plt.ylabel("Profit")
plt.show()

# ===================== HEATMAP =====================
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ===================== PAIRPLOT =====================
sns.pairplot(df[['Units Sold', 'Unit Price', 'Total Revenue', 'Total Profit']])
plt.show()

# ===================== BUBBLE PLOT =====================
plt.figure()
plt.scatter(df['Units Sold'], df['Total Revenue'],
            s=df['Total Profit']/10, alpha=0.5)
plt.title("Bubble Plot")
plt.xlabel("Units Sold")
plt.ylabel("Revenue")
plt.show()

# ===================== HYPOTHESIS TESTING =====================
# Compare profit between two regions
region1 = df[df['Region'] == df['Region'].unique()[0]]['Total Profit']
region2 = df[df['Region'] == df['Region'].unique()[1]]['Total Profit']

t_stat, p_val = ttest_ind(region1, region2)

print("T-Statistic:", t_stat)
print("P-Value:", p_val)

# ===================== STATISTICAL MODELING =====================
print("\nStatistical Summary:")
print(df.describe())

# ===================== LINEAR REGRESSION =====================
X = df[['Units Sold', 'Unit Price']]
y = df['Total Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\nLinear Regression R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ===================== ML MODEL (ADVANCED - MULTI FEATURE) =====================
X = df.drop('Total Profit', axis=1)
y = df['Total Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nML Model R2 Score:", r2_score(y_test, y_pred))
