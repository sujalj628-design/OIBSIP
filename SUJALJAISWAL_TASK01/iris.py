# Import Python Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading The Dataset
data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\my_project\OIBSIP\SUJALJAISWAL_TASK01\Iris.csv")
