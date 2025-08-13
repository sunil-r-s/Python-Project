import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🏠 House Price Prediction Model")
print("=" * 50)

try:
    # Load the dataset with error handling
    data = pd.read_csv(r'Python-Project\Python Project2\melb_data.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Original dataset shape: {data.shape}")
    
    # Basic info about the dataset
    print(f"📈 Columns in dataset: {list(data.columns)}")
    print(f"🔍 Missing values per column:")
    missing_info = data.isnull().sum()
    for col, missing in missing_info.items():
        if missing > 0:
            print(f"   {col}: {missing} ({missing/len(data)*100:.1f}%)")
    
    # Remove rows with missing values
    data_cleaned = data.dropna()
    print(f"🧹 After cleaning: {data_cleaned.shape}")
    print(f"📉 Rows removed: {data.shape[0] - data_cleaned.shape[0]}")
    
    # Show basic statistics
    print(f"\n📋 Dataset Overview:")
    print(data_cleaned.head())
    
    # Select features and target variable with more features
    features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt']
    
    # Check if all features exist in dataset
    available_features = []
    for feature in features:
        if feature in data_cleaned.columns:
            available_features.append(feature)
        else:
            print(f"⚠️  Feature '{feature}' not found in dataset")
    
    # Use available features
    X = data_cleaned[available_features]
    y = data_cleaned['Price']
    
    print(f"🎯 Selected features: {available_features}")
    print(f"📊 Feature statistics:")
    print(X.describe())
    
    # Check for any remaining missing values in selected features
    if X.isnull().sum().sum() > 0:
        print("🧹 Removing remaining missing values from features...")
        mask = X.isnull().any(axis=1) | y.isnull()
        X = X[~mask]
        y = y[~mask]
        print(f"📊 Final dataset shape: {X.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"🔄 Data split completed:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    
    # Create and train the model
    print("\n🤖 Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate multiple metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📈 Model Performance Metrics:")
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   Root Mean Square Error: ${rmse:,.2f}")
    print(f"   Mean Absolute Error: ${mae:,.2f}")
    
    # Feature importance (coefficients)
    print(f"\n🎯 Feature Importance (Coefficients):")
    feature_importance = dict(zip(available_features, model.coef_))
    for feature, coef in feature_importance.items():
        print(f"   {feature}: {coef:,.2f}")
    
    # Create enhanced visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue', s=30)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Prices ($)')
    axes[0, 0].set_ylabel('Predicted Prices ($)')
    axes[0, 0].set_title(f'Actual vs Predicted Prices\n(R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Prices ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature importance
    features_plot = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    axes[1, 0].barh(features_plot, importance_values, color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Coefficient Value')
    axes[1, 0].set_title('Feature Importance (Coefficients)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Prediction error distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example predictions
    print(f"\n🔮 Example Predictions:")
    print("-" * 40)
    sample_indices = np.random.choice(X_test.index, 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        actual = y_test.loc[idx]
        predicted = y_pred[y_test.index.get_loc(idx)]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        
        print(f"Example {i+1}:")
        print(f"  Features: {dict(X_test.loc[idx])}")
        print(f"  Actual Price: ${actual:,.2f}")
        print(f"  Predicted Price: ${predicted:,.2f}")
        print(f"  Error: ${error:,.2f} ({error_pct:.1f}%)")
        print()
    
    # Save the model with additional info
    model_info = {
        'model': model,
        'features': available_features,
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    joblib.dump(model_info, 'house_price_model_enhanced.pkl')
    print("✅ Enhanced model saved as 'house_price_model_enhanced.pkl'")
    
    # Function to make new predictions
    def predict_house_price(rooms, bathroom=None, landsize=None, building_area=None, year_built=None):
        """
        Predict house price based on features
        """
        # Create feature array based on available features
        feature_values = []
        feature_names = []
        
        if 'Rooms' in available_features:
            feature_values.append(rooms)
            feature_names.append('Rooms')
        if 'Bathroom' in available_features and bathroom is not None:
            feature_values.append(bathroom)
            feature_names.append('Bathroom')
        if 'Landsize' in available_features and landsize is not None:
            feature_values.append(landsize)
            feature_names.append('Landsize')
        if 'BuildingArea' in available_features and building_area is not None:
            feature_values.append(building_area)
            feature_names.append('BuildingArea')
        if 'YearBuilt' in available_features and year_built is not None:
            feature_values.append(year_built)
            feature_names.append('YearBuilt')
        
        # Make prediction
        prediction = model.predict([feature_values])[0]
        
        print(f"\n🏠 Price Prediction:")
        for name, value in zip(feature_names, feature_values):
            print(f"   {name}: {value}")
        print(f"   Predicted Price: ${prediction:,.2f}")
        
        return prediction
    
    # Example prediction
    print("\n" + "="*50)
    print("🎯 Example: Predicting price for a sample house")
    sample_prediction = predict_house_price(
        rooms=3,
        bathroom=2 if 'Bathroom' in available_features else None,
        landsize=600 if 'Landsize' in available_features else None,
        building_area=120 if 'BuildingArea' in available_features else None,
        year_built=2000 if 'YearBuilt' in available_features else None
    )

except FileNotFoundError:
    print("❌ Error: CSV file not found!")
    print("Please check the file path: 'Python-Project\\Python Project2\\melb_data.csv'")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")

print("\n" + "="*50)
print("🎉 Analysis Complete!")