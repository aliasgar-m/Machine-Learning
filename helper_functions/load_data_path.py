import pandas as pd
import os

def load_data(path=None, f_name=None):
    csv_path = os.path.join(path, f_name+".csv")
    return pd.read_csv(csv_path)