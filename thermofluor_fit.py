import os
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from math import ceil, floor
from scipy.optimize import curve_fit
from sklearn import preprocessing



def main():
    raw_df = pd.read_excel(input_file)
#     normalize min/max on y values
    norm_df = (raw_df - raw_df.min()) / (raw_df.max() - raw_df.min())
    x = norm_df.iloc[0:,0]
    y = norm_df.iloc[0:,1:]
    y_smooth = y.rolling(window = 5).mean()
    
    #1st derivative, smoothed
    smoothed_dfy = pd.concat([x, y_smooth.diff()], axis=1)
    smoothed_dfy = (smoothed_dfy - smoothed_dfy.min()) / (smoothed_dfy.max() 
                                                 - smoothed_dfy.min())
    
    #2nd derivative, smoothed
    ddfy = pd.concat([x, y_smooth.diff().diff()], axis=1)
    scaled_ddfy = (ddfy - ddfy.min()) / (ddfy.max() - ddfy.min())
    make_plots(norm_df, smoothed_dfy, scaled_ddfy, grouped=True, step=2)

def make_plots(y, dfy, ddfy, grouped=True, step=2):
    if not grouped and step != 1:
        raise ValueError(f'Cannot plot all separate graphs with step = {step}')
    if not grouped:
        datasets = y.shape[1]
        rows = int(ceil(datasets/4))
        columns = int(4)
    else:
        datasets = y.shape[1]
        rows = int(ceil(datasets/4/step))
        columns = int(4/step)
    fig, axes = plt.subplots(nrows=rows, ncols=columns)
    
    i = 1
    for ax in axes.reshape(-1):
        #TODO: figure out why the x axis is wrong
        ax.grid(True)
        try:
            for sub in range(step):
                t = i+sub
                _y = y.iloc[:,t]
                _dfy = dfy.iloc[:,i+sub]
                _ddfy = ddfy.iloc[:,i+sub] 
                py = _y.plot(ax=ax, color='blue', use_index=False)
                pdy = _dfy.plot(ax=ax, color='green', linestyle='dotted',  
                              use_index=False)
                pddy = _ddfy.plot(ax=ax, color='red', linestyle='dashed',  
                                use_index=False)
            i += step
        except IndexError: #more subplots than data available
            fig.delaxes(ax)
            
    fig.savefig('figure.png', figsize=(20,20), dpi=300, linewidth=0.0)
    plt.show()

def thermo_fit(T, m, n, r, hm, tm):
    R = 8.3144621 # J /K*mol
    baseline = (m*T + n) 
    K = (hm * (T-tm)) / (R*T*tm)
    num = baseline + np.exp(-r*T)*np.exp(K)
    den = 1 + np.exp(K)
    y = num / den
    return y

def logistic_fit(T, T0, k, L):
    y = L / (1 + np.exp(T-T0))
    return y 

if __name__ == '__main__':
    ### config ###
    working_dir = 'test/data'
    input_file = os.path.join(working_dir, 'tf_abs.xlsx')
    main()