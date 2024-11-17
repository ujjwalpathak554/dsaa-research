import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import itertools
import numpy as np

# Load the dataset (replace 'your_file.csv' with the actual file name)
file_path = 'dataset/Covid_Dataset.csv'  # Replace with your actual CSV file path
data = pd.read_csv(file_path)

# You may need to preprocess the data if required
# Example: If 'DATE_DIED' is not useful as a numeric feature, you might remove it
data = data.drop(columns=['DATE_DIED'], errors='ignore')

# Define the target variable (e.g., severity can be based on ICU, CLASIFFICATION_FINAL, or another column)
target = data['ICU']  # Example: 'ICU' as the target variable
features = data.drop(columns=['ICU'])

# Compute Mutual Information for each pair of features
mutual_info_results = {}
for feature1, feature2 in itertools.combinations(features.columns, 2):
    # Create a new DataFrame with just the two features
    feature_pair = features[[feature1, feature2]]
    
    # Compute mutual information with the target
    mi = mutual_info_classif(feature_pair, target, discrete_features=True)
    
    # Store the result
    mutual_info_results[(feature1, feature2)] = mi

# Display the results
for feature_pair, mi_values in mutual_info_results.items():
    print(f"Mutual Information between {feature_pair[0]} and {feature_pair[1]}: {mi_values}")

# If you want to save the results to a CSV file
mi_df = pd.DataFrame.from_dict(mutual_info_results, orient='index', columns=['MI_Feature1', 'MI_Feature2'])
mi_df.to_csv('mutual_information_results.csv')
