import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split

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

def binary_col(data):
    #return the binary columns
    binary_cols=[col for col in data.columns if data[col].nunique() == 2]
    return binary_cols

def continuous_col(data):
    #return the continuous columns
    continuous_cols=[col for col in data.columns if data[col].nunique() > 2]
    return continuous_cols
    

def Scaler(data):
    binary_cols = binary_col(data)
    continuous_cols = continuous_col(data)
    scaler = StandardScaler()
   # Scale the continuous columns
    scaled_continuous = scaler.fit_transform(data[continuous_cols])
    scaled_continuous_df = pd.DataFrame(scaled_continuous, columns=continuous_cols)
    # Combine the scaled continuous columns with the binary columns
    result = pd.concat([data[binary_cols].reset_index(drop=True), scaled_continuous_df.reset_index(drop=True)], axis=1)
    return result

#open file bnpl
data = pd.read_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl.csv')
data.info()

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    outliers = OutlierDetector(data[column])
    print(f"Outliers in {column}:\n", outliers)

data = DefaultFlagGenerator(data)
data_clean = data.drop(columns=['CustomerID'])

data_scaled = Scaler(data_clean)
data_scaled.to_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_scaled.csv', index=False)
print("Data processing complete. Scaled data saved to 'bnpl_scaled.csv'.")


# run histograms for continuous columns
continuous_cols = continuous_col(data_scaled)
binary_cols = binary_col(data_scaled)

#check for outliers in each column
for column in data_scaled.select_dtypes(include=['float64', 'int64']).columns:
    outliers = OutlierDetector(data_scaled[column])
    if not outliers.empty:
        print(f"Outliers in {column}:\n", outliers)
    else:
        print(f"No outliers detected in {column}.")


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

#drop the outliers from the data_scaled dataframe
data_scaled_cleaned = data_scaled.drop(index=outlier_indices)

#drop the over_indebtedness_flag column from the data_scaled_cleaned dataframe
data_scaled_cleaned = data_scaled_cleaned.drop(columns=['over_indebtedness_flag'])
binary_cols = [col for col in binary_cols if col != 'over_indebtedness_flag']
data_scaled_cleaned.to_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_scaled_cleaned.csv', index=False)


# concat the following stress_usage_interaction = financial_stress_score × bnpl_usage_frequency
data_scaled_cleaned['stress_usage_interaction'] = (
    data_scaled_cleaned['financial_stress_score'] * data_scaled_cleaned['bnpl_usage_frequency']
)

#adjusted_debt_interaction = bnpl_debt_ratio + financial_stress_score
data_scaled_cleaned['adjusted_debt_interaction'] = (
    data_scaled_cleaned['bnpl_debt_ratio'] * data_scaled_cleaned['financial_stress_score']
)
data_scaled_cleaned.to_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_scaled_cleaned_interactions.csv', index=False)

#get the mean and std of the new columns
new_columns = [
    'stress_usage_interaction',
    'adjusted_debt_interaction'
]
mean_std = data_scaled_cleaned[new_columns].agg(['mean', 'std'])
print("Mean and Standard Deviation of New Columns:")
print(mean_std)

results = []
for feat in new_columns:
    r, p = pointbiserialr(data_scaled_cleaned[feat], data_scaled_cleaned['default_flag'])
    results.append({'feature': feat, 'r': r, 'p_value': p})

corr_df = pd.DataFrame(results).sort_values('r', key=abs, ascending=False)
print(corr_df)

final_data = pd.read_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_scaled_cleaned_interactions.csv')

# 1) split off test set
train_val, test = train_test_split(
    final_data,
    test_size=0.20,
    stratify=final_data['default_flag'],
    random_state=42
)


# 2) split train vs. validation from the remaining 80%
train, val = train_test_split(
    train_val,
    test_size=0.25,                       # 0.25 × 80% → 20% overall
    stratify=train_val['default_flag'],
    random_state=42
)


print("Default rates:",
      train['default_flag'].mean(),
      val['default_flag'].mean(),
      test['default_flag'].mean())

train_idx = train.index.to_list()
val_idx   = val.index.to_list()
test_idx  = test.index.to_list()
#print the indices and the number of indices in each set
print(f"Train indices: {train_idx[:10]}... ({len(train_idx)} total)")
print(f"Validation indices: {val_idx[:10]}... ({len(val_idx)} total)")
print(f"Test indices: {test_idx[:10]}... ({len(test_idx)} total)")


#create 3 csv files for train, val and test sets
train.to_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_train.csv', index=False)
val.to_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_val.csv', index=False)
test.to_csv(r'C:\Users\ixz407\OneDrive - University of Birmingham\Dissertation\Machine Learning\Dissertation\Tables\bnpl_test.csv', index=False)
# Print the shape of each set
print(f"Train set shape: {train.shape}")
print(f"Validation set shape: {val.shape}")
print(f"Test set shape: {test.shape}")