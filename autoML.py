import pandas as pd 
import numpy as np 
import numpy as numpy

from sklearn import preprocessing

class autoML:
    def __init__(self, dataset, target, delimiter=None, drop_index = None):   
        self.dataset = dataset
        self.data = self.read_dataset(self.dataset, delimiter)
        self.target = target
        if drop_index != None:
            self.drop_index = drop_index

        self.processed = False

    def read_dataset(self, dataset, delimiter):
        return pd.read_csv(dataset, delimiter=delimiter)

    def process_data(self):
        """
        Drop index column if specified by user
        Drops columns with missing values
        """
        self.data = self.data.drop(self.drop_index, axis = 1)
        self.data = self.data.dropna(axis = 'columns')
    
    def show_data(self):
        print(self.data.head())

    def __str__(self):
        return f'{self.dataset} \nprocessed = {self.processed}'
    

if __name__ == "__main__":
    data = autoML('data.csv', drop_index='id', target = "diagnosis")
    data.process_data()
