# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:53:08 2020

@author: Vishal
"""
import pandas as pd

dataset = pd.read_csv('flink-release-1.7.csv')                  #Enter dataset name here

dataset = pd.DataFrame(dataset)

data = dataset[dataset['Kind'].str.contains('Class') == True]

data = data[['Kind', 'Name', 'CountLine' ,'CountLineCode', 'CountLineBlank', 'CountLineComment', 
             'CountClassBase', 'CountClassCoupled', 'CountClassDerived', 'CountDeclClassMethod', 
             'CountDeclClassVariable', 'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 
             'CountDeclMethod', 'CountDeclMethodAll', 'CountDeclMethodDefault',
             'CountDeclMethodPrivate', 'CountDeclMethodProtected', 'CountDeclMethodPublic', 
             'MaxInheritanceTree', 'PercentLackOfCohesion']]

data.to_csv('flink-release-1.7-preprocessed.csv')    


