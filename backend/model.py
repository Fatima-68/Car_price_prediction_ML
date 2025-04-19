import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib

# Load data
df = pd.read_csv("car_prices.csv")

# Minimal useful columns
df = df[['year','make','model','trim', 'body', 'transmission', 'state', 'condition', 'odometer','seller', 'mmr', 'sellingprice']]
df = df.dropna()

X = df.drop("sellingprice", axis=1)
y = df["sellingprice"]

# Categorical and numerical
cat_cols = ['make','model','trim','body', 'transmission', 'state','seller',]
num_cols = ['year', 'condition', 'odometer', 'mmr']

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# Model pipeline with XGBoost
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Save
joblib.dump(model, "best_model.pkl")
print(" XGBoost model trained and saved as best_model.pkl")
