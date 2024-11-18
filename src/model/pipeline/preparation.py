'''Organize all data related to data preparation. Data processing functions.'''

import pandas as pd
from collection import load_data

def prepare_data():
    data = load_data()
    return data