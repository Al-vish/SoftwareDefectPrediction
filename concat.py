# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:26:54 2019

@author: spand
"""

import pandas as pd

d1 = pd.read_csv("GEODE-8324-preprocessed.csvchange.csv")
d1.fillna(0)	
d2 = pd.read_csv("GEODE-8444-preprocessed.csvchange.csv")
d2.fillna(0)
d3 = pd.read_csv("GEODE-8483-preprocessed.csvchange.csv")
d3.fillna(0)
#d4 = pd.read_csv("4.csvchange.csv")


#l = [d1, d2, d3, d4]
l = [d1, d2, d3]

data = pd.concat(l)


data.to_csv("GeodFeinal.csv")