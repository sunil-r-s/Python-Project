import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Python-Project/Temp_data.csv'
df = pd.read_csv(file_path)


#Available column names (to debug if needed)
print('Available columns:', df.columns.tolist())

#Display first 5 rows
print("🔍 First 5 Rows:")
print(df.head())

#Summary statisticss
print("\n📊 Summary Statistics:")
print(df.describe())

#Check for null values
print("\n🧼 Null Values in Each Column:")
print(df.isnull().sum())


#Calculate average temperature
average_temp = df['Temperature'].mean()
print(f"\n🌡️ Average Temperature: {average_temp:.2f}°C")

#Group average temperature per city for bar chart
city_avg_temp = df.groupby('City')['Temperature'].mean().reset_index()

#Bar chart: Average temperature per city
plt.figure(figsize=(6,4))
city_avg_temp.plot(kind='bar')
plt.title('Average Temperature per City')
plt.xlabel('City')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

#Scatter plot: Temperature vs Humidity
plt.figure(figsize=(6,4))
plt.scatter(df['Temperature'], df['Humidity'], color='green')
plt.title('Temperature vs Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

#Heatmap: Correlation between numeric  columns
plt.figure(figsize=(6,4))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

#Insights and observations
print("\n📌 Insights:")
print("1. The average temperature across all cities is {:.2f}°C.".format(average_temp))
print("2. Bar chart shows which city is hotter or cooler on average.")
print("3. Scatter plot may show whether higher temperatures lead to higher/lower Humidity.")
print("4. Heatmap reveals how temperature, humidity, and wind speed are related.")