import os
import pandas as pd

def read_and_append_csv_in_folder(folder_path, output_file):
    dataframes = []
    
    # Iterate through all the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            try:
                # Try reading the file as a CSV into a dataframe
                df = pd.read_csv(file_path)
                dataframes.append(df)
            except pd.errors.ParserError:
                # If the file cannot be read as a CSV, skip it
                print(f"File {filename} is not in CSV format, skipping.")
    
    # Concatenate all dataframes in the list into a single dataframe
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"All CSV format files have been combined into {output_file}.")
    else:
        print("No CSV format files found.")


read_and_append_csv_in_folder('Address')