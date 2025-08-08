import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv(r'Python-Poject2\melb_data.csv')
print("Before Cleaning:", data.shape)

# Remove rows with missing Values
data = data.dropna()
print("After Cleaning:", data.shape)

#show the first few rows
print(data.head())

#Select features and target variable
features = ['Rooms', 'Landsize', 'BuildingArea']
X = data[features]
y = data['Price']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Check model accuracy (R² score)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy (R² score): {accuracy:.2f}")

# Predict prices on the test set
y_pred = model.predict(X_test)


# Plot predicticted vs actual prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()


# Save the trained model to file
import joblib
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl")