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

    information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)
    
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
    for column in df.columns.tolist()[:-1]:
        igDict[column] = float(get_information_gain(df, column))

    selected_attribute = max(igDict, key=igDict.get)
    ansTuple = (igDict, selected_attribute)

    return ansTuple
