import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("sales_data.csv")

# Features and target
X = df[['Month']]
y = df['Sales']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict future sales
future_months = pd.DataFrame({
    'Month': [9, 10, 11, 12]
})

predictions = model.predict(future_months)

# Print predictions
print("Future Sales Predictions:\n")

for month, sale in zip(future_months['Month'], predictions):
    print(f"Month {month}: {sale:.2f}")

# Plot original data
plt.scatter(df['Month'], df['Sales'], label="Actual Sales")

# Plot prediction line
plt.plot(
    future_months['Month'],
    predictions,
    label="Predicted Sales"
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Prediction")

plt.legend()
plt.grid(True)

plt.show()