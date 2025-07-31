import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
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


data = pd.read_csv('bnpl.csv')

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    outliers = OutlierDetector(data[column])
    print(f"Outliers in {column}:\n", outliers)


data = DefaultFlagGenerator(data)
data_clean = data.drop(columns=['CustomerID'])

data_scaled = Scaler(data_clean)
data_scaled.to_csv('bnpl_scaled.csv', index=False)
print("Data processing complete. Scaled data saved to 'bnpl_scaled.csv'.")

continuous_cols = continuous_col(data_scaled)
binary_cols = binary_col(data_scaled)

#print histograms for continuous columns
for column in continuous_cols:
    plt.figure(figsize=(10, 6))
    plt.hist(data_scaled[column], bins=30, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'histograms/{column}_histogram.png')
    plt.close()
    

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
    plt.figtext(0.5, 0.01, f"Balance of {column}: {balance.to_dict()}", ha='center', fontsize=10)
    plt.tight_layout()
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

#drop the outliers from the data_scaled dataframe
data_scaled_cleaned = data_scaled.drop(index=outlier_indices)

#plot 2 boxplots for each column one with default flag 0 and one with default flag 1
continuous_cols = continuous_col(data_scaled_cleaned)
for column in continuous_cols:
    plt.figure(figsize=(10, 6))
    data_scaled_cleaned.boxplot(column=column, by='default_flag')
    plt.title(f'Boxplot of {column} by Default Flag')
    plt.suptitle('')
    plt.xlabel('Default Flag')
    plt.ylabel(column)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'boxplots/{column}_boxplot_by_default_flag.png')
    plt.close()
    
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
plt.close()

#what is the numerical value of the corr of 'bnpl_usage_frequency' and 'over_indebtedness_flag'
corr_value = correlation_matrix_cleaned.loc['bnpl_usage_frequency', 'over_indebtedness_flag']
print(f"Correlation between 'bnpl_usage_frequency' and 'over_indebtedness_flag': {corr_value:.4f}")

#drop the over_indebtedness_flag column from the data_scaled_cleaned dataframe
data_scaled_cleaned = data_scaled_cleaned.drop(columns=['over_indebtedness_flag'])
binary_cols = [col for col in binary_cols if col != 'over_indebtedness_flag']
data_scaled_cleaned.to_csv('bnpl_scaled_cleaned.csv', index=False)


# concat the following stress_usage_interaction = financial_stress_score × bnpl_usage_frequency
data_scaled_cleaned['stress_usage_interaction'] = (
    data_scaled_cleaned['financial_stress_score'] * data_scaled_cleaned['bnpl_usage_frequency']
)

#adjusted_debt_interaction = bnpl_debt_ratio + financial_stress_score
data_scaled_cleaned['adjusted_debt_interaction'] = (
    data_scaled_cleaned['bnpl_debt_ratio'] * data_scaled_cleaned['financial_stress_score']
)
data_scaled_cleaned.to_csv('bnpl_scaled_cleaned_interactions.csv', index=False)

#get the mean and std of the new columns
new_columns = [
    'stress_usage_interaction',
    'adjusted_debt_interaction'
]
mean_std = data_scaled_cleaned[new_columns].agg(['mean', 'std'])
print("Mean and Standard Deviation of New Columns:")
print(mean_std)

#plot a new correlation matrix with the new columns
correlation_matrix_new = data_scaled_cleaned.corr()
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix_new, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Matrix with New Columns')
#change the x and y ticks to the column names
plt.xticks(ticks=np.arange(len(data_scaled_cleaned.columns)), labels=data_scaled_cleaned.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(data_scaled_cleaned.columns)), labels=data_scaled_cleaned.columns)
plt.show()

final_data = pd.read_csv('bnpl_scaled_cleaned_interactions.csv')

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
train.to_csv('train_set.csv', index=False)
val.to_csv('val_set.csv', index=False)
test.to_csv('test_set.csv', index=False)