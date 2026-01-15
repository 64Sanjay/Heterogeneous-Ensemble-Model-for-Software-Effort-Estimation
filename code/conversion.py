import pandas as pd

# Load the CSV file to examine its contents
file_path = './nasadataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
df.head()

# Defining the mapping based on the provided reference values
value_mapping = {
    'acap': {'vl': 1.46, 'l': 1.19, 'n': 1.00, 'h': 0.86, 'vh': 0.71, 'xh': 2.06},
    'pcap': {'vl': 1.42, 'l': 1.17, 'n': 1.00, 'h': 0.86, 'vh': 0.70, 'xh': 1.67},
    'aexp': {'vl': 1.29, 'l': 1.13, 'n': 1.00, 'h': 0.91, 'vh': 0.82, 'xh': 1.57},
    'modp': {'vl': 1.24, 'l': 1.10, 'n': 1.00, 'h': 0.91, 'vh': 0.82, 'xh': 1.34},
    'tool': {'vl': 1.24, 'l': 1.10, 'n': 1.00, 'h': 0.91, 'vh': 0.83, 'xh': 1.49},
    'vexp': {'vl': 1.21, 'l': 1.10, 'n': 1.00, 'h': 0.90, 'vh': None, 'xh': 1.34},
    'lexp': {'vl': 1.14, 'l': 1.07, 'n': 1.00, 'h': 0.95, 'vh': None, 'xh': 1.20},
    'sced': {'vl': 1.23, 'l': 1.08, 'n': 1.00, 'h': 1.04, 'vh': 1.10, 'xh': None},
    'stor': {'vl': None, 'l': None, 'n': 1.00, 'h': 1.06, 'vh': 1.21, 'xh': 1.56},
    'data': {'vl': None, 'l': 0.94, 'n': 1.00, 'h': 1.08, 'vh': 1.16, 'xh': None},
    'time': {'vl': None, 'l': None, 'n': 1.00, 'h': 1.11, 'vh': 1.30, 'xh': 1.66},
    'turn': {'vl': None, 'l': 0.87, 'n': 1.00, 'h': 1.07, 'vh': 1.15, 'xh': None},
    'virt': {'vl': None, 'l': 0.87, 'n': 1.00, 'h': 1.15, 'vh': 1.30, 'xh': None},
    'rely': {'vl': 0.75, 'l': 0.88, 'n': 1.00, 'h': 1.15, 'vh': 1.40, 'xh': None},
    'cplx': {'vl': 0.70, 'l': 0.85, 'n': 1.00, 'h': 1.15, 'vh': 1.30, 'xh': 1.65}
}

# Replace the categorical values in the dataset with corresponding numerical values
for column in value_mapping.keys():
    if column in df.columns:
        df[column] = df[column].map(value_mapping[column])

# Display the updated dataframe to verify changes
df.head()


# Drop the specified columns: 'recordnumber', 'projectname', 'year'
columns_to_drop = ['recordnumber', 'projectname', 'year','cat2','forg','center','mode']
df_cleaned = df.drop(columns=columns_to_drop)

df_cleaned.head()

df_cleaned.to_csv("revised_dataset.csv",index=False)
