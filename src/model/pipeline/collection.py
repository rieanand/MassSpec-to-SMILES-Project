'''Loads data required. Handles loading of our data sets.'''
import pandas as pd

def load_data(path):
    return pd.read_csv(path)