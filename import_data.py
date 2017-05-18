import os
import glob
import pandas as pd

def main():
    os.chdir(working_dir)
    #read file#
    files = glob.glob('*.csv')
    with open(files[0], 'rb') as f:
        df = pd.read_csv(f, encoding = "ISO-8859-1")
    a = df.iloc[0:,1:2]
    print(a)
    

if __name__ == '__main__':
    ####### Config #######
    working_dir = os.path.abspath('test/data')
    main()