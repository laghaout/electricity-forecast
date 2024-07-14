# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:06:21 2024

@author: amine
"""

import os
from dotenv import load_dotenv
import json
import logging
import os
import pandas as pd
import pydantic
import sys
from types import SimpleNamespace

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def read_csv(directory, file, **kwargs):
    
    match file.split('.')[-1].lower():
        case 'csv':
            data = pd.read_csv(os.path.join(directory, file))
        case _:
            assert False, f'Invalid file {file}'
    
    return data

def disp(text=None):
    
    logging.info(text)

def load_json_as_dict(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        disp(f"Directory '{directory}' created.")
    else:
        disp(f"Directory '{directory}' already exists.")
        
def get_env_variables(env_file='.env'):
    # Load environment variables from a .env file
    load_dotenv(env_file)

    # Retrieve all environment variables and store them in a dictionary
    env_vars = {key: value for key, value in os.environ.items()}
    env_vars = SimpleNamespace(**env_vars)
    
    return env_vars        

def compute_correlations(df, target_column):
    """
    Computes the correlation between a specified target column and all other columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column to compute correlations with.

    Returns:
    pd.DataFrame: A DataFrame containing the correlation coefficients.
    """
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame.")
    
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    
    # Extract the correlations with the target column
    target_correlations = correlation_matrix[[target_column]].drop(target_column)
    
    return target_correlations
