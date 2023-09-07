import pandas as pd
import re
from prettytable import PrettyTable


class SourceTypeSeparator:
    def __init__(self, fundamental_dataset_dir, first_col_id, tiles_names, split_craters_by_tile_dir):
        # Where is CSV dataset:
        self.fundamental_data_dir = fundamental_dataset_dir
        # Variables to proceed it properly
        self.first_col_id = first_col_id
        self.tiles_names = tiles_names
        # Where to save to CSV files:
        self.split_craters_by_tile_dir = split_craters_by_tile_dir

    def split_craters_by_tile_id(self):
        # Read the big CSV file
        df = pd.read_csv(self.fundamental_data_dir)

        # Define the list of specific ID patterns with digit placeholders
        id_patterns = self.tiles_names.keys()

        # Loop through each ID pattern and filter rows
        for pattern in id_patterns:
            filtered_rows = df[df[self.first_col_id].str.match(pattern)]

            # Write the filtered rows to a new CSV file
            output_file = f'{self.split_craters_by_tile_dir}\\{self.tiles_names[pattern]}.csv'
            filtered_rows.to_csv(output_file, index=False)

    def analyze_split_crater_by_tile_id(self, cols_names_to_analyze):
        myTable = PrettyTable(["FILE_NAME"] + cols_names_to_analyze)

        id_patterns = self.tiles_names.keys()
        for pattern in id_patterns:
            input_file_dir = f'{self.split_craters_by_tile_dir}\\{self.tiles_names[pattern]}.csv'
            df = pd.read_csv(input_file_dir)
            cols_span = []
            for col_name in cols_names_to_analyze:
                min_val = round(df[col_name].min(), 2)
                max_val = round(df[col_name].max(), 2)
                avg_val = round(df[col_name].mean(), 2)
                med_val = round(df[col_name].median(), 2)
                cols_span.append(f"MIN:{min_val}, MAX:{max_val}, AVG:{avg_val}, MED:{med_val}")
            myTable.add_row([self.tiles_names[pattern]] + cols_span)
        print(myTable)