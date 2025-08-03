import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Define utility functions
def OutlierDetector(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data < lower) | (data > upper)]

def DefaultFlagGenerator(df):
    default_flag = (
        (df['payment_delinquency_count'] >= 3).astype(int) +
        (df['over_indebtedness_flag'] == 1).astype(int) +
        (df['financial_stress_score'] >= 9).astype(int) +
        (df['bnpl_debt_ratio'] >= 1.8).astype(int) +
        (df['credit_limit_utilisation'] >= 95).astype(int)
    ) >= 3
    df['default_flag'] = default_flag.astype(int)
    return df

def binary_col(data):
    return [col for col in data.columns if data[col].nunique() == 2]

def continuous_col(data):
    return [col for col in data.columns if data[col].nunique() > 2]

def Scaler(data):
    binary_cols = binary_col(data)
    continuous_cols = continuous_col(data)
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(data[continuous_cols])
    scaled_continuous_df = pd.DataFrame(scaled_continuous, columns=continuous_cols)
    result = pd.concat([data[binary_cols].reset_index(drop=True), scaled_continuous_df.reset_index(drop=True)], axis=1)
    return result

# Step 2: Define the machine learning pipeline
def ml_pipeline(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Detect and print outliers
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        outliers = OutlierDetector(data[column])
        print(f"Outliers in {column}:\n", outliers)

    # Generate default flag
    data = DefaultFlagGenerator(data)
    data_clean = data.drop(columns=['CustomerID'])

    # Scale data
    data_scaled = Scaler(data_clean)
    data_scaled.to_csv('bnpl_scaled.csv', index=False)
    print("Data processing complete. Scaled data saved to 'bnpl_scaled.csv'.")

    # Detect and remove outliers
    continuous_cols = continuous_col(data_scaled)
    outlier_indices = []
    for column in continuous_cols:
        outliers = OutlierDetector(data_scaled[column])
        outlier_indices.extend(outliers.index.tolist())
    data_scaled_cleaned = data_scaled.drop(index=outlier_indices)

    # Add interaction features
    data_scaled_cleaned['stress_usage_interaction'] = (
        data_scaled_cleaned['financial_stress_score'] * data_scaled_cleaned['bnpl_usage_frequency']
    )
    data_scaled_cleaned['adjusted_debt_interaction'] = (
        data_scaled_cleaned['bnpl_debt_ratio'] * data_scaled_cleaned['financial_stress_score']
    )
    data_scaled_cleaned.to_csv('bnpl_scaled_cleaned_interactions.csv', index=False)

    # Split data into train, validation, and test sets
    train_val, test = train_test_split(
        data_scaled_cleaned,
        test_size=0.20,
        stratify=data_scaled_cleaned['default_flag'],
        random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val['default_flag'],
        random_state=42
    )

    # Save splits to CSV
    train.to_csv('train_set.csv', index=False)
    val.to_csv('val_set.csv', index=False)
    test.to_csv('test_set.csv', index=False)

    # Train a machine learning model
    X_train = train.drop(columns=['default_flag'])
    y_train = train['default_flag']
    X_val = val.drop(columns=['default_flag'])
    y_val = val['default_flag']
    X_test = test.drop(columns=['default_flag'])
    y_test = test['default_flag']

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_val = model.predict(X_val)
    print("Validation Set Performance:")
    print(confusion_matrix(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val))

    y_pred_test = model.predict(X_test)
    print("Test Set Performance:")
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

# Step 3: Run the pipeline
if __name__ == "__main__":
    ml_pipeline('bnpl.csv')