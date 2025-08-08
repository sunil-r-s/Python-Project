import joblib
import pandas as pd

# Load saved model from file
model = joblib.load('house_price_model.pkl')

# Example input: [Rooms, Landsize, BuildingArea]
new_data = pd.DataFrame([[3,250,120]], columns=['Rooms', 'Landsize', 'BuildingArea'])

# Predict the price
predicted_price = model.predict(new_data)
print(f"Predicted house price: ${predicted_price[0]:,.2f}")