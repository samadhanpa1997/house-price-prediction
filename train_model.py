import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("house_data.csv")

#Check null records or missing values
print(df.isnull().sum())

#Check duplicate records
print(df.drop_duplicates())

# Create EDA (understanding relationships between our features and house price.)

plt.figure(figsize=(8,5))
plt.scatter(df["area"], df["price"])

plt.xlabel("area")
plt.ylabel("price")
plt.title("Area vs House Price")

# plt.show()

# Features and Target
X = df[["area", "bedrooms", "bathrooms","age"]]
y = df["price"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Actual House Price", y_test)
print("Predicted House Price", y_pred)

#Comparison
df = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print(df)
# Evaluation
#print("R2 Score:", r2_score(y_test, y_pred))
#print("MAE:", mean_absolute_error(y_test, y_pred))

# Coefficients
"""print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef}")"""

#print("Intercept:", model.intercept_)

# Example Prediction

#Test With New data
new_house=pd.DataFrame({
    "area": [1500],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [40],
})

predicted_price = model.predict(new_house)
print("Predicted House Price", predicted_price)

#Save Model
with open("house_price_prediction.pkl", "wb") as file:
    pickle.dump(model, file)

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()