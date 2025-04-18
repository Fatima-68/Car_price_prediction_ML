import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load your CSV
data = pd.read_csv("car_prices.csv")

data = data.drop(['make', 'vin', 'color'], axis=1)

# Dropping the duplicates 
data = data.drop_duplicates()
# 🧼 Drop rows with missing target values
data = data.dropna(subset=['sellingprice']) 
data =data.dropna()

# Select only numeric columns for IQR calculation
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Now compute quantiles and IQR only on numeric columns
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers based on IQR
data = data[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
# Define target and features
X = data.drop("sellingprice", axis=1)
y = data["sellingprice"]

# Optionally drop rows with missing features (or handle them better later)
X = X.dropna()

# Identify categorical & numerical columns
categorical = X.select_dtypes(include=["object"]).columns
numerical = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')
data = data.sample(100, random_state=42)
# Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=5, random_state=42))
])

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data prepared")
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

model.fit(X_train, y_train)

print("✅ Model training complete")
# Save the model
joblib.dump(model, "best_model.pkl")
print("✅ Model trained and saved as best_model.pkl")
