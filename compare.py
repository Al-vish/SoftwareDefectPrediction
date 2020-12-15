# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:59:17 2019

@author: spand
"""
import numpy as np
import pandas as pd

columns = ['Kind','Name','CountLineCode','CountClassBase', 'CountClassCoupled',
       'CountClassDerived', 'CountDeclMethodAll', 'CountDeclMethod',
       'MaxInheritanceTree', 'PercentLackOfCohesion',
       'CountDeclInstanceMethod', 'CountDeclInstanceVariable']

metrics = ['CountLineCode','CountClassBase', 'CountClassCoupled',
       'CountClassDerived', 'CountDeclMethodAll', 'CountDeclMethod',
       'MaxInheritanceTree', 'PercentLackOfCohesion',
       'CountDeclInstanceMethod', 'CountDeclInstanceVariable']

# Note:
# I first compared NL, if there was a difference I compared coupling, 
# inheritance tree and cohesion metrics, if they showed improvement in 
# the next version, and the code lines were not increased, there was a
# fault that was corrected, hence 1, otherwise 0.

# Need low IT, low coupling, high cohesion

def func(version1,version2) :

    # Read version 1 and sanitize
    csv1 = pd.read_csv(version1)
    csv1 = csv1[columns] 
    csv1.dropna(inplace=True)
    csv1 = csv1[csv1.CountLineCode > 0]
    
    # Read version 2 and sanitize
    csv2 = pd.read_csv(version2)
    csv2 = csv2[columns]
    csv2.dropna(inplace=True)
    csv2 = csv2[csv2.CountLineCode > 0]
    
    # Get the names of classes
    names = csv1.Name.tolist()

    # list of our target variable
    ans = []

    # For each class
    for i in names :
        fault_corrected = 0
        v2 = csv2[csv2.Name == i]
        if not v2.empty :
            v1 = csv1[csv1.Name == i]
            # If the lines of code are non-increasing
            if v1.iloc[0]['CountLineCode'] >= v2.iloc[0]['CountLineCode']:
                # and if DIT has become lower
                if v1.iloc[0]['MaxInheritanceTree'] > v2.iloc[0]['MaxInheritanceTree']:
                    fault_corrected = 1
                # or if coupling has decreased
                elif v1.iloc[0]['CountClassCoupled'] > v2.iloc[0]['CountClassCoupled']:
                    fault_corrected = 1
                # or if cohesion has increased
                elif v1.iloc[0]['PercentLackOfCohesion'] > v2.iloc[0]['PercentLackOfCohesion']:
                    fault_corrected = 1
        
        # If the class has been deleted, mark fault as corrected
        else:
            fault_corrected = 1 
        
        ans.append(fault_corrected)   
    
    # Add a new column Change to the v1 file
    csv1['Change'] = ans
    
    # save the file to disk
    csv1.to_csv(ver1 + "change.csv")
    

ver1 = 'flink-release-1.4-preprocessed.csv' #Change the path to version 1
ver2 = 'flink-release-1.5-preprocessed.csv' #Change the path to version 2

func(ver1,ver2)