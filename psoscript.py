# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:54:29 2020

@author: Vishal
"""
import sys
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Import PySwarms
import pyswarms as ps
df = pd.read_csv("preprocessedv1.4.csv")
X = df.data
y = df.target


from sklearn.model_selection import train_test_split



import numpy as np