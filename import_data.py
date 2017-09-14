import os
import glob
import pandas as pd
import numpy as np
import xlsxwriter as xls #needed for pd.ExcelWriter
import openpyxl as op
from xlsxwriter.utility import xl_rowcol_to_cell
# from openpyxl.cell.cell import 

def conv(x):
    try:
        x = float(x)
        return x
    except ValueError:
        return x
    
def main():
    os.chdir(working_dir)
    #read file#
    files = [f for f in glob.glob('*.csv') if '_vs_Temp' in f]
    #setup xls writer
    writer = pd.ExcelWriter(f'{outfile}', engine='xlsxwriter')
    workbook = writer.book
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    
    for input_ in files:
        with open(input_, 'rb') as f:
            df = pd.read_csv(f, encoding = "ISO-8859-1")
        df.applymap(conv)
        sheet_title = df.columns[1].split()[0]
        run = str(int(df.iloc[1,1].replace('run','')))
        wells = [f'{sheet_title}_{i}' for i in list(df.iloc[2,]) if 'Well' not in i]
        wells = [f'Run {run}'] + wells
        #remove all Temperature columns except the first
        _,cols = np.where( df=='Temperature') 
        df.drop(df.columns[cols[1:]], axis=1, inplace=True) 
        
        #remove pointless rows and cleanup
        df = df.iloc[3:,]
        df.drop(df.index[1], inplace=True)
        df.iloc[0,0] = 'Temperature'
        df.columns = wells
        df.to_excel(writer, sheet_name=f'{sheet_title}')

    last_row, last_column = df.shape #all datasets are the same in size, hopefully
    
    writer.save()
    
    #changing data type of value cells with openpyxl
    #otherwise excel will think that this is text and refuse to graph

    op_wb = op.load_workbook(outfile)
    for worksheet in op_wb:
        for row in worksheet[3:last_row+1]:
            for cell in row[1:]:
                cell.data_type = 'n'
    op_wb.save(outfile) 
    op_wb.close()   
    
    #now we reopen things with xlsxWriter to make all the charts
    wb = writer.book
    sheets = []
    for s in wb.worksheets():
        sheets.append(s.get_name()) #otherwise it dynamically updates during for 
    for sheet_name in sheets: 
        first_row = 2
        first_column = 1
        x_values = [f'{sheet_name}', first_row, first_column, last_row, first_column]
        _x_values = xl_rowcol_to_cell(first_row,first_column)
        __x_values = xl_rowcol_to_cell(last_row,first_column)
        print(_x_values, __x_values)
        for col in range(2, last_column+1):
            y_values = [f'{sheet_name}', first_row, col,
                                         last_row, col]
            _y_values = xl_rowcol_to_cell(first_row, col)
            __y_values = xl_rowcol_to_cell(last_row, col)
            name = xl_rowcol_to_cell(0,col)
            print(name)
            chart = workbook.add_chart({'type':'scatter',
                                        'subtype':'straight_with_markers'})
            chart.add_series({'name' : f'=${sheet_name}!{name}',
                              'categories' : f'=${sheet_name}!{_x_values}{__x_values}',
                              'values': f'=${sheet_name}!{_y_values}{__y_values}'
            })
        data = [10, 40, 50, 20, 10, 50]
        worksheet = wb.add_worksheet()
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'values': '=Sheet1!$A$1:$A$6'})
        worksheet.insert_chart('C1', chart)
        workbook.close()
#         wb.get_worksheet_by_name(sheet_name).insert_chart('P1',chart)
#         chartsheet = workbook.add_chartsheet(f'{sheet_name}_graph')
#         chartsheet.set_chart(chart)
    writer.save()
    wb.close()

if __name__ == '__main__':
    ####### Config #######
    working_dir = os.path.abspath('/home/andrea/Desktop/optim')
    print(os.path.isdir(working_dir))
    outfile = os.path.join(working_dir, 'out.xlsx')
    main()