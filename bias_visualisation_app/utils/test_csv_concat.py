import pandas as pd
import os
from os import listdir
from os.path import isfile, join


def concat_csv_excel():
    path_parent = os.path.dirname(os.getcwd())
    csv_path = os.path.join(path_parent, 'static', 'user_downloads')
    writer = pd.ExcelWriter(os.path.join(csv_path, 'complete_file.xlsx'))  # Arbitrary output name
    csvfiles = [f for f in listdir(csv_path) if isfile(join(csv_path, f))]
    print(csvfiles)
    for csvfilename in csvfiles:
        df = pd.read_csv(os.path.join(csv_path, csvfilename))
        df.to_excel(writer, sheet_name=os.path.splitext(csvfilename)[0])
    writer.save()

concat_csv_excel()