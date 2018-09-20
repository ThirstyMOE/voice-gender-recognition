import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

source_df = pd.read_csv("voice.csv")
print(source_df.head())
