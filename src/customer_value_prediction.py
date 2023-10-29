import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

train_raw = pd.read_csv('./src/data/train_customer_segmented.csv', sep=',')
val_raw = pd.read_csv('./src/data/validation_customer_segmented.csv', sep=',')
test_raw = pd.read_csv('./src/data/test_customer_segmented.csv', sep=',')

