import os
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from math import ceil, floor
from scipy.optimize import curve_fit
from sklearn import preprocessing



def main(order=False, normalize=False, plot_y=True, plot_dy=True, plot_ddy=True,
         make_plots=True, step=2, remove_wells=0):
    
    df = pd.read_excel(input_file)
    
    # normalize min/max on y values
    if normalize:
        df = _normalize(df)
    
    if make_plots:
        #order the dataframe per well
        if order and type(order) == type([]):
            y = reorder_wells(df, order)
        elif order and not type(order) == type([]):
            msg = 'if you wish to reorder the columns, pass a list of well names to the function'
            raise ValueError(msg)
        else:
            y = df
            order = list(y.columns)
        
        #remove unwanted columns at the end
        if remove_wells:
            keep = [i for i in order if i not in remove_wells]
            y = y[keep]
        #smooth data
        y_smooth = y.rolling(window = 5).mean()
        #calculate 1st derivative, smoothed
        dy = y_smooth.diff()
        if normalize:
            dy = _normalize(dy)    
        #calculate 2nd derivative, smoothed
        ddy = dy.diff()
        if normalize:
            ddy = _normalize(ddy)
        
        #choose what to plot
        if not plot_y:
            y = None
        if not plot_dy:
            dy=None
        if not plot_ddy:
            ddy=None
        
        #make the actual plots
        generate_plots(y, dy, ddy, grouped=True, step=step)

def _normalize(df):
    
    df = (df - df.min()) / (df.max() - df.min())
    return df

def generate_plots(y=None, dy=None, ddy=None, grouped=True, step=2):
    
    #check that there is something to plot
    try:
        a = next(i for i in [y,dy,ddy] if i is not None)
    except StopIteration:
        print('Nothing to plot')
        return
    
    if not grouped and step != 1: #internal reminder
        raise ValueError(f'Cannot plot all separate graphs with step = {step}')
    
    #get shape of graph grid
    datasets = a.shape[1]
    if datasets/step <= 4:
        columns = int(datasets/step)
    else:
        columns = 4
        
    if not grouped:
        rows = int(ceil(datasets/4))
    else:
        rows = ceil(datasets/4/step)
        
    fig, axes = plt.subplots(nrows=rows, ncols=columns, squeeze=False) #squeeze = False => always return a 2d np.array(), even for ncols=1, nrows=1
    
    y_cmap = plt.get_cmap('Blues')
    dy_cmap = plt.get_cmap('Greens')
    ddy_cmap = plt.get_cmap('Reds')
    y_line_colors = y_cmap(np.linspace(0,1,step))
    dy_line_colors = dy_cmap(np.linspace(0,1,step))
    ddy_line_colors = ddy_cmap(np.linspace(0,1,step))
    
    i = 0
    for ax in axes.flatten(): #axes.reshape(-1) only works if multiple subplots are available)
        ax.grid(True)
        ax.sharex = True
        titles = [] 
        try:
            for sub in range(step):
                t = i+sub
                title = a.iloc[:,t].name
                titles.append(title)
                if y is not None: #df truth is ambiguous in pandas
                    _y = y.iloc[:,t]
#                     py = _y.plot(ax=ax, color='blue', use_index=True)
                    py = _y.plot(ax=ax, color=y_line_colors[sub], use_index=True,
                                 label=f'Int ({title})')
                if dy is not None:
                    _dy = dy.iloc[:,i+sub]
                    pdy = _dy.plot(ax=ax, x = dy.index, color=dy_line_colors[sub],
                     linestyle='dotted')
                if ddy is not None:
                    _ddy = ddy.iloc[:,i+sub] 
                    pddy = _ddy.plot(ax=ax, color=ddy_line_colors[sub],
                                      linestyle='dashed')
            title = ''.join(f'{t} ' for t in titles)
            ax.set_title(title)
#             legend = ax.legend(loc='upper right', shadow=True)
            i += step
        except IndexError: #more subplots than data available => delete subplot
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
    input_file = os.path.join(working_dir, 'ferritin_repro_test_conc_series.xlsx')
    #specifyng replicates to graph together
    order = []
    
    letters = ['A','B','C','D','E','F','G','H']
    first_well = 1
    last_well = 6
    normalize=1
    make_plots = 1 #debug of program workflow
    plot_y=1
    plot_dy=0
    plot_ddy=1
    remove_wells = 0
    step=6
     
    #duplicate position - uncomment relevant one
    #A1-B1, A2-B2, etc.
#     for t in [['A','B'], ['C','D'], ['E','F'], ['G','H']]:
#         for n in range(1,7):
#             for l in t:#,'C','D','E','F','G','H']:
#                 n = str(n).zfill(2)
#                 order.append(f'{l}{n}')
#     step=2

#     #6 duplicates on a line
    order = []
    for l in letters:
        for n in range(first_well,last_well+1):
            n = str(n).zfill(2)
            order.append(f'{l}{n}')
    step=6
    
    #remove lines 
#     lines_to_remove = []
#     remove_wells = []
#     for l in lines_to_remove:
#         for n in range(first_well,last_well+1):
#             n = str(n).zfill(2)
#             remove_wells.append(f'{l}{n}')
#     step=6
    
    
    main(order=order, normalize=normalize, plot_y=plot_y, plot_dy=plot_dy,
         plot_ddy=plot_ddy, make_plots=make_plots, step=step, 
         remove_wells=remove_wells)