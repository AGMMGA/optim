import os
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from math import ceil, floor
from scipy.optimize import curve_fit
from sklearn import preprocessing



def main(order=False, normalize=False, plot_y=True, plot_dy=True, plot_ddy=True):
    
    df = pd.read_excel(input_file)
    if normalize:
#     normalize min/max on y values
        df = (df - df.min()) / (df.max() - df.min())
    if order and type(order) == type([]):
        y = reorder_wells(df, order)
    elif order and not type(order) == type([]):
        msg = 'if you wish to reorder the columns, pass a list of well names to the function'
        raise ValueError(msg)
    else:
        y = df
    y_smooth = y.rolling(window = 5).mean()
    #1st derivative, smoothed
    dy = y_smooth.diff()
    if normalize:
        dy = (dy - dy.min()) / (dy.max() - dy.min())    
    #2nd derivative, smoothed
    ddy = dy.diff()
    if normalize:
        ddy = (ddy - ddy.min()) / (ddy.max() - ddy.min())
    if not plot_y:
        y = None
    if not plot_dy:
        dy=None
    if not plot_ddy:
        ddy=None
    make_plots(y, dy, ddy, grouped=True, step=2)

def make_plots(y=None, dy=None, ddy=None, grouped=True, step=2):
    try:
        a = next(i for i in [y,dy,ddy] if i is not None)
    except StopIteration:
        print('Nothing to plot')
        return
    if not grouped and step != 1:
        raise ValueError(f'Cannot plot all separate graphs with step = {step}')
    if not grouped:
        datasets = a.shape[1]
        rows = int(ceil(datasets/4))
        columns = int(4)
    else:
        datasets = a.shape[1]
        rows = int(ceil(datasets/4/step))
        columns = int(4/step)
    fig, axes = plt.subplots(nrows=rows, ncols=columns)
    
    i = 0
    for ax in axes.reshape(-1):
        #TODO: figure out why the x axis is wrong
        ax.grid(True)
        ax.sharex = True
        titles = [] 
        try:
            for sub in range(step):
                t = i+sub
                titles.append(a.iloc[:,t].name)
                if y is not None:
                    _y = y.iloc[:,t]
                    py = _y.plot(ax=ax, color='blue', use_index=True)
                if dy is not None:
                    _dy = dy.iloc[:,i+sub]
                    pdy = _dy.plot(ax=ax, x = ddy.index, color='green', linestyle='dotted')
                if ddy is not None:
                    _ddy = ddy.iloc[:,i+sub] 
                    pddy = _ddy.plot(ax=ax, color='red', linestyle='dashed')
            title = ''.join(f'{t} ' for t in titles)
            ax.set_title(title)
            i += step
        except IndexError: #more subplots than data available
            fig.delaxes(ax)
    plt.subplots_adjust(hspace=0.5)
#     fig.savefig('figure.png', figsize=(20,20), dpi=300, linewidth=0.0)
    plt.show()

def reorder_wells(unordered_frame, custom_order):
    '''
    custom_order: the ordered list of column labels
    must contain all the labels present in the unordered data frame,
    just in different order
    e.g. 
    unordered_frame.columns == ['A','B','C']
    custom_order == ['B','C','A']
    '''
    columns = list(unordered_frame.columns)
    if not custom_order.copy().sort() == columns.sort(): #copy() otherwise custom_order gets sorted in place ;-)
        msg =f'''
        Custom order not valid; some wells are missing'
        Custom order : {custom_order}
        Unordered frame labels: {columns}
        '''
        raise ValueError(msg)
    f = unordered_frame.copy()
    f.columns = custom_order
    return f

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
    input_file = os.path.join(working_dir, 'ferritin.xlsx')
    #specifyng replicates to graph together
    order = []
    for t in [['A','B'], ['C','D'], ['E','F'], ['G','H']]:
        for n in range(1,7):
            for l in t:#,'C','D','E','F','G','H']:
                n = str(n).zfill(2)
                order.append(f'{l}{n}')
    normalize=True
    plot_y=False
    plot_dy=True
    plot_ddy=True
    
    main(order=order, normalize=normalize, plot_y=plot_y, plot_dy=plot_dy,
         plot_ddy=plot_ddy)