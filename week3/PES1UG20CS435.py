'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the entire dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    # TODO

    # getting the entropy of each column
    columnEntropies = []
    for column in df.columns:
        vc = df[column].value_counts(normalize=True, sort=False)
        entropy1 = -(vc * np.log(vc)/np.log(2)).sum()
        columnEntropies.append(entropy1)

    # getting the entropy of attribute to be classified in the decision tree (last entry)
    entropy = columnEntropies[len(columnEntropies) - 1]

    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO

    encodings = df[[attribute, df.columns[-1]]]
    uniqueElements = df[attribute].unique().tolist()
    
    sumEntropies = 0
    for x in uniqueElements:
        frac = df[attribute].to_list().count(x) / len(df[attribute])
        sumEntropies += get_entropy_of_dataset(encodings[encodings[attribute] == x]) * frac

    avg_info = sumEntropies

    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO

    data = df
    split_name = attribute

    # calculating the original entropy
    original_entropy = get_entropy_of_dataset(df)
    
    # finding the unique values in the column
    values = data[split_name].unique()
    
    # making two subsets of the data, based on the unique values
    left_split = data[data[split_name] == values[0]]
    right_split = data[data[split_name] == values[1]]
    
    # looping through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * get_entropy_of_dataset(df)
    
    # calculating information gain
    information_gain = original_entropy - to_subtract
    
    return information_gain


#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    
    igDict = dict()
    for column in df.columns:
        igDict[column] = get_information_gain(df, column)

    ansTuple = (igDict, max(igDict, key=igDict.get))
    print(ansTuple[0])

    return ansTuple
