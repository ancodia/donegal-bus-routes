from openpyxl import load_workbook
import pandas as pd

townload_population_xls = '../data/population/COP2016_Townlands.xlsx'

if __name__ == '__main__':
    # load excel data
    population_data = load_workbook(townload_population_xls)

    # convert to pandas dataframe, set first row as header
    df = pd.DataFrame(population_data.worksheets[0].values).T.set_index(0).T
    # filter so only Donegal townlands remain
    df = df[df['COUNTY'] == 'DL']
    # combine address into a single column and
    # keep only relevant data, i.e. place name and 2016 census figure
    df['address'] = df['TLANDNAME'] + ', ' + \
                    df['EDNAMES_3409S'] + ', ' + \
                    df['COUNTYNAME']
    df = df[['address', 'TOTAL2016']]
    df.to_csv('../data/population/donegal_townlands.csv')
