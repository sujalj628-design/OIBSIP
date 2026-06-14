# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading The Dataset
data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\my_project\OIBSIP\SUJALJAISWAL_TASK02\Advertising.csv")

print("++++++ SALES PREDICTION USING PYTHON ++++++\n")

# Dataset Exploration AND FEATURE SELECTION
print("Dataset Shape:", data.shape)

print("\nCOLUMNS IN DATASET:")
print(data.columns)

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
