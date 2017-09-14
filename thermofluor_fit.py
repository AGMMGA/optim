import os
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from math import ceil, floor
from scipy.optimize import curve_fit
from sklearn import preprocessing



def main(order=False, normalize=False, plot_y=True, plot_dy=True, plot_ddy=True,
         make_plots=True, step=2, remove_wells=0, cols=4, delete=[]):
    
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
        #remove unwanted columns
        if remove_wells:
            keep = [i for i in order if i not in remove_wells]
            y = y[keep]
        #remove unwanted wells by adding zeroes instead
        for well in delete:
            y[well] = np.zeros((y.shape[0], 1))
        #smooth data
        y_smooth = y.rolling(window = 1).mean()
        #calculate 1st derivative, smoothed
        dy = y_smooth.diff()
        if normalize:
            dy = _normalize(dy)
            dy=dy/2.5 #makes for easier read   
        #calculate 2nd derivative, smoothed
        ddy = dy.diff()
        if normalize:
            ddy = _normalize(ddy)
            ddy = ddy/2.5
        #choose what to plot
        if not plot_y:
            y = None
        if not plot_dy:
            dy=None
        if not plot_ddy:
            ddy=None
        
        #make the actual plots
        if draw_plots:
            generate_plots(y, dy, ddy, grouped=True, step=step, cols=cols, delete=delete)

def _normalize(df):
    
    df = (df - df.min()) / (df.max() - df.min())
    return df

def generate_plots(y=None, dy=None, ddy=None, grouped=True, step=2, cols=4, delete=[]):
    
    #check that there is something to plot
    try:
        a = next(i for i in [y,dy,ddy] if i is not None)
    except StopIteration:
        print('Nothing to plot')
        return
    
    if not grouped and step != 1: #internal reminder
        raise ValueError(f'Cannot plot all separate graphs with step = {step}')
    
    #get shape of graph grid
    datasets = a.shape[1] -len(delete_curves)
    if datasets/step <= cols:
        columns = int(datasets/step)
    else:
        columns = cols
    if not grouped:
        rows = int(ceil(datasets/cols))
    else:
        rows = ceil(datasets/cols/step)
    fig, axes = plt.subplots(nrows=rows, ncols=columns, squeeze=False)  #squeeze = False => always return a 2d np.array(), even for ncols=1, nrows=1
    
    y_cmap = plt.get_cmap('gist_rainbow')
    dy_cmap = plt.get_cmap('Blues')
    ddy_cmap = plt.get_cmap('Reds')
    y_line_colors = y_cmap(np.linspace(0,1,step))
    dy_line_colors = dy_cmap(np.linspace(0.5,0.75,step))
    ddy_line_colors = ddy_cmap(np.linspace(0.25,0.75,step))
    
    i = 0
    for ax in axes.flatten(): #axes.reshape(-1) only works if multiple subplots are available)
        ax.grid(True)
        ax.sharex = True
        titles = [] 
        try:
            for sub in range(step):
                t = i+sub
                title = a.iloc[:,t].name
                if title in delete: #we have chosen not to plot this series
                    continue
                titles.append(title)
                melt_temps = []
                if y is not None: #df truth is ambiguous in pandas
                    _y = y.iloc[:,t]
#                     py = _y.plot(ax=ax, color='blue', use_index=True)
                    py = _y.plot(ax=ax, color=y_line_colors[sub], use_index=True,
                                 label=title)
                if dy is not None:
                    _dy = dy.iloc[:,i+sub]
                    pdy = _dy.plot(ax=ax, x = dy.index, color=dy_line_colors[sub],
                     linestyle='dashed')
                    melt_temp = max(dy)
                if ddy is not None:
                    _ddy = ddy.iloc[:,i+sub] 
                    pddy = _ddy.plot(ax=ax, color=ddy_line_colors[sub],
                                      linestyle='dotted')
            title = ''.join(f'{t} ' for t in titles)
#             title = title + (f'Tm (avg) = {np.average(melt_temps)}')
            ax.set_title(title)
            if show_legend:
                legend = ax.legend(loc='upper right', shadow=True)
            i += step
        except IndexError: #more subplots than data available => delete subplot
            fig.delaxes(ax)
    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(20,10)
    fig.tight_layout()
    if save_figure:        
        fig.savefig(figure_name, dpi=300, linewidth=0.0)
    if show_plot:
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
    f = f[custom_order]
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

def group_by(grouping):
    order = []
    valid_groupings = ['rows', 'columns', 'a_b', '1_2']
    if grouping not in valid_groupings:
        msg = f'Unrecognised grouping type: {grouping}.\n Valid options are "lines", "columns", "a_b", "1_2"'
        raise ValueError(msg)
    elif grouping == 'columns':
        for n in range (first_well, last_well+1):
            for l in letters:
                n = str(n).zfill(2)
                order.append(f'{l}{n}')
        step = 8
        cols = 6
    elif grouping == 'rows':
        for l in letters:
            for n in range(first_well,last_well+1):
                n = str(n).zfill(2)
                order.append(f'{l}{n}')
        step = 6
        cols = 8
    elif grouping == 'a_b': #(A1,B1; C1,D1, etc.)
        for t in range(1,7):
            for n in [['A','B'], ['C','D'], ['E','F'], ['G','H']]:
                for l in n:
                    order.append(f'{l}{str(t).zfill(2)}')
        step = 2
        cols = 4
    elif grouping == '1_2': #(A1,A2; A3,A4; etc.)
        for t in [['A','B'], ['C','D'], ['E','F'], ['G','H']]:
            for n in range(1,7):
                for l in t:
                    n = str(n).zfill(2)
                    order.append(f'{l}{n}')
        step = 2
        cols = 6
    return order, step, cols

if __name__ == '__main__':
    ### config ###
    working_dir = 'test/data'
    input_file = os.path.join(working_dir, 'Complex_mix.xlsx')
    #specifyng replicates to graph together
    letters = ['A','B','C','D','E','F','G','H']
    first_well = 1
    last_well = 6
    normalize=0
    make_plots = 1 #debug of program workflow
    draw_plots = 1 #debug of program workflow
    plot_y=1
    plot_dy=0
    plot_ddy=0
    remove_wells = False
    step=1
    save_figure = 0
    figure_name = 'Complex_mix_05.png'
    show_plot = 1
    show_legend = 1
    
    #valid options: rows, columns, 1_2 (A1,A2; A3,A4; etc.), a_b (A1,B1; C1,D1, etc.)
    order, step, cols = group_by('a_b')
    delete_curves = []
    #uncomment to delete curves
#     delete_curves = ['A06','B06','C06','D06', 'E06','F06', 'H06', 'G06']
    
    
    main(order=order, normalize=normalize, plot_y=plot_y, plot_dy=plot_dy,
         plot_ddy=plot_ddy, make_plots=make_plots, step=step, 
         remove_wells=remove_wells, cols=cols, delete=delete_curves)