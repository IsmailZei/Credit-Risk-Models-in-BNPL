import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def OutlierDetector(data):
    '''Detects outliers in a given pandas Series using the IQR method.
    Args:
        data (pd.Series): The input data series to check for outliers.
    Returns:
        pd.Series: A series containing the outliers.
    Raises:
        ValueError: If the input data is not a pandas Series.   
    
    '''
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return data[(data < lower) | (data > upper)]

def Scaler(data, continuous_cols, binary_cols):
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(data[continuous_cols])
    scaled_continuous_df = pd.DataFrame(scaled_continuous, columns=continuous_cols, index=data.index)
    
    # Combine binary columns and scaled continuous columns
    result = pd.concat([data[binary_cols], scaled_continuous_df], axis=1)
    
    return result
    

def DefaultFlagGenerator(df):
    default_flag = (
        (df['payment_delinquency_count'] >= 3).astype(int) +
        (df['over_indebtedness_flag'] == 1).astype(int) +
        (df['financial_stress_score'] >= 9).astype(int) +
        (df['bnpl_debt_ratio'] >= 1.8).astype(int) +
        (df['credit_limit_utilisation'] >= 95).astype(int)
        ) >= 3 # Must meet at least 3 of the 5 conditions
    df['default_flag'] = default_flag.astype(int)
    return df

data = pd.read_csv('bnpl.csv')

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
binary_cols = [col for col in num_cols if data[col].nunique() == 2]
continuous_cols = [col for col in num_cols if data[col].nunique() > 2]

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    outliers = OutlierDetector(data[column])
    print(f"Outliers in {column}:\n", outliers)
    
data = DefaultFlagGenerator(data)
data_clean = data.drop(columns=['CustomerID'])
data_scaled = Scaler(data_clean, continuous_cols, binary_cols)

# Assuming 'data_scaled' is a NumPy array and 'df' is your original DataFrame
# Replace the continuous columns in the original DataFrame with the scaled data

# Save the scaled DataFrame to a CSV file
data_scaled.to_csv('bnpl_scaled.csv', index=False)

print("Data processing complete. Scaled data saved to 'bnpl_scaled.csv'.")

# run histograms for continuous columns
for column in continuous_cols:
    plt.figure(figsize=(10, 6))
    plt.hist(data_scaled[column], bins=30, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'histograms/{column}_histogram.png')
    plt.close()
    

#check for outliers in each column
for column in data_scaled.select_dtypes(include=['float64', 'int64']).columns:
    outliers = OutlierDetector(data_scaled[column])
    if not outliers.empty:
        print(f"Outliers in {column}:\n", outliers)
    else:
        print(f"No outliers detected in {column}.")
    
#print boxplots for continuous columns
for column in continuous_cols:
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_scaled[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'boxplots/{column}_boxplot.png')
    plt.close()
    
#check the balance of the binary columns
for column in binary_cols:
    balance = data_scaled[column].value_counts(normalize=True)
    print(f"Balance of {column}:\n", balance)
    plt.figure(figsize=(8, 5))
    balance.plot(kind='bar')
    plt.title(f'Balance of {column}')
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'balances/{column}_balance.png')
    plt.close()

# Initialize an empty list to store outlier indices
outlier_indices = []

# Check for outliers in the scaled data by splitting the data into two parts: one with default_flag = 0 and one with default_flag = 1
data_default_0 = data_scaled[data_scaled['default_flag'] == 0]
data_default_1 = data_scaled[data_scaled['default_flag'] == 1]

for column in continuous_cols:
    outliers_0 = OutlierDetector(data_default_0[column])
    outliers_1 = OutlierDetector(data_default_1[column])
    
    # Add indices of outliers to the list
    outlier_indices.extend(outliers_0.index.tolist())
    outlier_indices.extend(outliers_1.index.tolist())
    
    if not outliers_0.empty:
        print(f"Outliers in {column} for default_flag = 0:\n", outliers_0)
    else:
        print(f"No outliers detected in {column} for default_flag = 0.")
    
    if not outliers_1.empty:
        print(f"Outliers in {column} for default_flag = 1:\n", outliers_1)
    else:
        print(f"No outliers detected in {column} for default_flag = 1.")

# Print the list of outlier indices
print("Outlier indices:", outlier_indices)

data_scaled_cleaned = data_scaled.drop(index=outlier_indices)

#plot the correlation matrix again after removing outliers
correlation_matrix_cleaned = data_scaled_cleaned.corr()
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix_cleaned, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Matrix (Cleaned Data)')
#change the x and y ticks to the column names
plt.xticks(ticks=np.arange(len(data_scaled_cleaned.columns)), labels=data_scaled_cleaned.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(data_scaled_cleaned.columns)), labels=data_scaled_cleaned.columns)
plt.savefig('correlation_matrix_cleaned.png')

#what is the numerical value of the corr of 'bnpl_usage_frequency' and 'over_indebtedness_flag'
corr_value = correlation_matrix_cleaned.loc['bnpl_usage_frequency', 'over_indebtedness_flag']
print(f"Correlation between 'bnpl_usage_frequency' and 'over_indebtedness_flag': {corr_value:.4f}")


#drop the over_indebtedness_flag column from the data_scaled_cleaned dataframe
data_scaled_cleaned = data_scaled_cleaned.drop(columns=['over_indebtedness_flag'])
binary_cols = [col for col in binary_cols if col != 'over_indebtedness_flag']
data_scaled_cleaned.to_csv('bnpl_scaled_cleaned.csv', index=False)
print("Data processing complete. Cleaned scaled data saved to 'bnpl_scaled_cleaned.csv'.")