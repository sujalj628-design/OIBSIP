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