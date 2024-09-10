import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
plt.style.use('fivethirtyeight')

def plot_features_importances(importances:list, names:list, title='', path_to=None) -> None:
    """
    Plot features importances for XGboost or RandomForest models
    """
    fig, ax = plt.subplots(figsize=(15,8))
    plt.title(title)
    ax.bar(names, importances)
    plt.xticks(rotation=90)
    if path_to:
        plt.savefig(path_to)
    plt.show()
    return None

def plot_generation_vs_irradiance(generation:pd.Series, irradiance:pd.Series, title='', path_to=None) -> None:
    """
    Plot a line plot with Generation and Irradiance in different yaxis
    """
    fig, ax = plt.subplots(figsize=(15,8))
    ax2 = ax.twinx()

    ax2.plot(generation, color='k', label='generation')
    ax2.set_ylabel('Generation (MW)')


    ax.plot(irradiance, color='b', label='irradiance')
    ax.set_ylabel('Irradiance (W/m²)')
    plt.legend()
    plt.title(title, weight='bold')

    if path_to:
        plt.savefig(path_to)
    plt.show()
    return None


def plot_generation_predicted_vs_observed(predicted:pd.Series, observed:pd.Series, title:str = '', path_to:str = None) -> None:
    """
    Plot a line plot with Generation and Irradiance in different yaxis
    """
    fig, ax = plt.subplots(figsize=(15,8))

    ax.plot(observed, color='k', label='observed')
    ax.plot(predicted, color='b', label='predicted')
    

    ax.set_ylabel('Generation (MW)')
    plt.legend()
    plt.title(title, weight='bold')

    if path_to:
        plt.savefig(path_to)
    plt.show()   
    return None  