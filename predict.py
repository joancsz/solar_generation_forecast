from config import config
from src.data_process import load_features_dataframe, negative_to_zero, bias_removal
from src.visualization import plot_generation_vs_irradiance

import joblib
import pandas as pd
import numpy as np
from datetime import date

cfg = config['train']

def main():

    data = load_features_dataframe()
    model = joblib.load('model/model.joblib')
    prediction = model.predict(data)

    prediction = negative_to_zero(prediction)
    prediction = pd.Series(index=data.index, data=prediction, name='generation')
    prediction = bias_removal(prediction)
    data = pd.concat([data, prediction], axis=1)

    today = date(2024,6,15)
    today_str = today.strftime("%Y_%m_%d")
    today_str_for_plot = today.strftime("%Y-%m-%d")

    data.to_excel(f'forecasts/forecast_{today_str}.xlsx')

if __name__ == '__main__':
    main()