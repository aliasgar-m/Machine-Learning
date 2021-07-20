import pandas as pd
import os
import sys
sys.path.append("/home/ali/Machine-Learning")
from helper_functions import fetch_data, load_data

ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
URL = ROOT + "datasets/housing/housing.tgz"
D_NAME = "housing"
PATH = os.path.join(os.getcwd(),"datasets")

try:
    dataset = load_data(path=PATH, f_name=D_NAME)
except:
    fetch_data(f_url=URL, d_name=D_NAME)
    dataset = load_data(path=PATH, f_name=D_NAME)

print(dataset.head(5).to_string(), '\n')
print(dataset.info(), '\n')
print(dataset['ocean_proximity'].value_counts())
