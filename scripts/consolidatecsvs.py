import os
import pandas as pd

# This script consolidates all the CSVs in the dataset into one file for easier lookups
# Only needs to be run once
if __name__ == '__main__':
    PATH = "data/dataset"
    dataframes = []

    for file in os.listdir(PATH):
        if file.endswith('.csv'):
            file_path = os.path.join(PATH, file)
            print(f'file name: {file}')
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    consolidated_df = pd.concat(dataframes, ignore_index = True)
    consolidated_df.to_csv('data/consolidated.csv', index = False)
    print(f'Consolidated {len(dataframes)} CSVs.')