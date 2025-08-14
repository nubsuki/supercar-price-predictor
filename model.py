# Libraries
import pandas as pd
import numpy as np
import joblib
import json

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
df = pd.read_csv("supercars_train.csv")
print ("Dataset Shape:", df.shape)

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

# SAVE THE MODEL
print("\nSaving the trained model...")
joblib.dump(final_pipe, 'supercar_price_model.pkl')

# Save feature information
feature_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'best_model': best_name,
    'model_performance': results
}

with open('model_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f" Model saved as 'supercar_price_model.pkl'")
print(f" Model info saved as 'model_info.json'")
print(f" Best model: {best_name} (RMSE: {best_row['RMSE']:,.0f}, R²: {best_row['R2']:.3f})")

# Load test set
try:
    test_df = pd.read_csv("supercars_test.csv")
    
    # Keep ID
    if 'ID' in test_df.columns:
        test_ids = test_df['ID']
    elif 'id' in test_df.columns:
        test_ids = test_df['id']
    else:
        print("Warning: Test set must contain an 'ID' or 'id' column for submission")
        test_ids = range(len(test_df))

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
    print("\n Submission file created: submission.csv")
    print(submission.head())
    
except FileNotFoundError:
    print("\nNote: Test file not found. Skipping submission creation.")