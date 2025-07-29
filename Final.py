import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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

def Scaler(data):
    scaler = StandardScaler()
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    binary_cols = [col for col in num_cols if data[col].nunique() == 2]
    continuous_cols = [col for col in num_cols if data[col].nunique() > 2]
   # Scale the continuous columns
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

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    outliers = OutlierDetector(data[column])
    print(f"Outliers in {column}:\n", outliers)
    
data = DefaultFlagGenerator(data)
data_clean = data.drop(columns=['CustomerID'])
data_scaled = Scaler(data_clean)

# Assuming 'data_scaled' is a NumPy array and 'df' is your original DataFrame
# Replace the continuous columns in the original DataFrame with the scaled data

# Save the scaled DataFrame to a CSV file
data_scaled.to_csv('bnpl_scaled.csv', index=False)

print("Data processing complete. Scaled data saved to 'bnpl_scaled.csv'.")

