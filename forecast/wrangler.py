# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:35:49 2024

@author: amine
"""

from pydantic import BaseModel, Extra
import forecast.utilities as util
import pandas as pd
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

class Wrangler(BaseModel):

    dataset: object
    
    class Config:
        extra = Extra.allow 
    
    def __call__(self):
        
        self.dataset.dropna(subset=[self.target], inplace=True)
        self.dataset.sort_values(by=self.sort_by, inplace=True)
        self.dataset.drop(self.drop, axis=1, inplace=True)
        self.dataset.dropna(axis=1, how='any', inplace=True)
        self.dataset[self.target] = self.dataset[self.target].shift(self.shift)
        self.features = [feature for feature in self.dataset.columns
                    if feature != self.target]
        # TODO: Add moving averages here.
        # Scale the features
        self.dataset = self.split(
            self.dataset, self.test_size, self.random_state)        

        return self
    
    @staticmethod
    def split(df, test_size, random_state):
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state,
            shuffle=isinstance(random_state, int))
        return SimpleNamespace(**dict(train=train, test=test))