# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
df = pd.read_csv("/content/supercars_train.csv")
print ("Dataset Shape:", df.shape)
df.head()

# Inspection
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribtion of Supercars")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Horsepower vs price
plt.figure(figsize=(8,5))
sns.scatterplot(x='horsepower', y='price', hue='brand', alpha=0.6, legend=False, data=df)
plt.title("Horsepower vs Price")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()

# Top brands by average price
top_brands = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=top_brands.index, y=top_brands.values)
plt.xticks(rotation=45)
plt.title("Top 10 Brands by Average Price")
plt.ylabel("Average Price")
plt.show()

# Correlation heatmap
plt.figure(figsize=(20,15))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Data preprocessing

# fill missing values in dmg columns
if 'damage_cost' in df.columns:
    df['damage_cost'] = df['damage_cost'].fillna(0)
if 'damage_type' in df.columns:
    df['damage_type'] = df['damage_type'].fillna('none')

# convert last service data to days since service
if 'last_service_date' in df.columns:
    df['last_service_date'] = pd.to_datetime(df['last_service_date'], errors='coerce')
    latest_date = df['last_service_date'].max()
    df['days_since_service'] = (latest_date - df['last_service_date']).dt.days
    df.drop(columns=['last_service_date'], inplace=True)

# drop id column
df.drop(columns=['id'], inplace=True, errors='ignore')

# features x and target y
X = df.drop(columns=['price'])
y = df['price']

# identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Build the model

num_transformer = SimpleImputer(strategy='mean')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ]
)

# Models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "LinearRegression": LinearRegression(),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist"),
    "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

# Train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    print(f"\nTraining {name}...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    results.append({"Model": name, "RMSE": rmse, "R2": r2})
    fitted_pipelines[name] = pipe

# results table
res_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\nModel Performance (sorted by RMSE):")
print(res_df.to_string(index=False, formatters={"RMSE": "{:,.0f}".format, "R2": "{:.3f}".format}))

# bar plot of RMSE
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="RMSE", data=res_df)
plt.title("RMSE by Model (lower is better)")
plt.xticks(rotation=20)
plt.ylabel("RMSE") # Root mean square deviation
plt.show()

# pick best model
best_row = res_df.iloc[0]
best_name = best_row["Model"]
best_pipe = fitted_pipelines[best_name]
print(f"\nBest Model: {best_name}")

# Refit best model on full training data
final_model = models[best_name]
final_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', final_model)])
final_pipe.fit(X, y)

# Load test set
test_df = pd.read_csv("/content/supercars_test.csv")

# Keep ID
if 'ID' in test_df.columns:
    test_ids = test_df['ID']
elif 'id' in test_df.columns:
    test_ids = test_df['id']
else:
    raise ValueError("Test set must contain an 'ID' or 'id' column")

# Apply same preprocessing
if 'damage_cost' in test_df.columns:
    test_df['damage_cost'] = test_df['damage_cost'].fillna(0)
if 'damage_type' in test_df.columns:
    test_df['damage_type'] = test_df['damage_type'].fillna('none')
if 'last_service_date' in test_df.columns:
    test_df['last_service_date'] = pd.to_datetime(test_df['last_service_date'], errors='coerce')
    latest_date = test_df['last_service_date'].max()
    test_df['days_since_service'] = (latest_date - test_df['last_service_date']).dt.days
    test_df.drop(columns=['last_service_date'], inplace=True)

# Drop ID column for prediction
X_submit = test_df.drop(columns=[c for c in ['ID', 'id'] if c in test_df.columns])

# Predict
test_preds = final_pipe.predict(X_submit)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'price': test_preds
})
# Save
submission.to_csv("submission.csv", index=False)
print("\nSubmission file created: submission.csv")
print(submission.head())

