import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def remove_unwanted_col(df, limit=0.5):
    """
    This function removes unwanted columns from a dataframe based on a threshold of NaN values
    :param df: The dataframe to remove unwanted columns
    :param limit: the percentage of NaN's in a column
    :return: New dataset
    """
    thr = limit * len(df)
    clean_df = df.dropna(axis=1, thresh=thr)
    return clean_df