# Import python Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading The Dataset
data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\my_project\OIBSIP\SUJALJAISWAL_TASK03\car data.csv")

print("++++++ CAR PRICE PREDICTION ++++++\n")

# DATASET EXPLORATION
print("Dataset Shape:", data.shape)

print("\nColumns:")
print(data.columns)

# CONVERTING TEXT COLUNM INTO NUMBER
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Selling_type'] = data['Selling_type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

# FEARTURE SELECTION
X = data[['Present_Price', 'Driven_kms', 'Fuel_Type',
            'Selling_type', 'Transmission', 'Owner']]

y = data['Selling_Price']

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# TRANING MODELAND PERFORMANCE
model = LinearRegression()
model.fit(X_train, y_train)

print("\nCAR PRICE PREDICTION \n")

present_price = float(input("Enter Present Price (in lakhs): "))
driven_kms = int(input("Enter Driven Kilometers: "))
fuel_type = int(input("Enter Fuel Type (Petrol=0, Diesel=1, CNG=2): "))
selling_type = int(input("Enter Selling Type (Dealer=0, Individual=1): "))
transmission = int(input("Enter Transmission (Manual=0, Automatic=1): "))
owner = int(input("Enter Number of Previous Owners: "))

user_data = [[
    present_price,
    driven_kms,
    fuel_type,
    selling_type,
    transmission,
    owner
]]

prediction = model.predict(user_data)
print("\nPREDICTED SELLING PRICE:", round(prediction[0], 2), "Lakhs")
