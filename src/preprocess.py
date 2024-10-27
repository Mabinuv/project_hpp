import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def remove_unwanted_col(df, limit=0.5):
    thr = limit * len(df)
    clean_df = df.dropna(axis=1, thresh=thr)
    return clean_df