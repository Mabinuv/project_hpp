import pandas as pd
import numpy as np
import os
import argparse

from matplotlib.pyplot import xcorr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def remove_unwanted_col(df, limit=0.5):
    """
    This function removes unwanted columns from a dataframe based on a threshold of NaN values
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - limit (float): The percentage threshold for NaN values; columns with NaNs above this limit are removed.
    Returns:
    - pd.DataFrame: DataFrame with columns containing a percentage of NaNs above the threshold removed.
    """
    thr = limit * len(df)
    clean_df = df.dropna(axis=1, thresh=thr)
    return clean_df

def handle_num(df):
    """
    Handling numerical columns :- Fills missing columns with the mean of each column
    :parameter:- The input DataFrame.
    :return:
    - pd.DataFrame: Modified DataFrame
    """
    numerical_col = df.select_dtypes(include=[int, float]).columns
    df.loc[:, numerical_col] = df[numerical_col].fillna(df[numerical_col].mean())
    return df

def handle_category(df):
    """
        Fills missing values in categorical columns with the mode and applies label encoding.
        Parameters:
        - Dataframe: The input DataFrame.
        Returns:
        - pd.DataFrame: Modified DataFrame
        """
    category_col = df.select_dtypes(include=['object']).columns
    for col in category_col:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    for col in category_col:
        df.loc[:, col] = label_encoder.fit_transform(df[col])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess a dataset for house price prediction.")
    parser.add_argument("input_file", type=str, help="Path to the input raw dataset CSV file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="The test size for train_test split (default is 0.2).")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input_file)
    raw_df = remove_unwanted_col(raw_df, limit=0.55) ## Remove col with 55% of NaN's value
    raw_df = handle_num(raw_df)     ## Handling numerical col with mean values
    raw_df = handle_category(raw_df) ## handling categorical values with model and label encoding

    ## Splitting and normalizing dataset for training
    X = raw_df.drop('SalePrice', axis=1)
    y = raw_df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Scale the features
    sc = StandardScaler()
    X_train_scaled = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Combine features and target for train and test sets
    train_df = pd.concat([X_train_scaled, y_train], axis=1)
    test_df = pd.concat([X_test_scaled, y_test], axis=1)

    # Save the processed dataframe to data/
    save_folder = "/Users/mabin/PycharmProjects/Project/HPP/data"
    os.makedirs(save_folder, exist_ok=True)
    train_df.to_csv(os.path.join(save_folder, 'train_df.csv'), index=False)
    test_df.to_csv(os.path.join(save_folder, 'test_df.csv'), index=False)
    print("Processed Train and test datasets saved in the data/*.")

