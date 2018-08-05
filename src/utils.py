import pandas as pd
from pathlib import Path

def create_raw_data_df_list(dir_path):
    list_files = searching_all_files(dir_path)
    list_files_names = [x._str.split('/')[-1] for x in list_files]
    df_imgs_paths = pd.DataFrame.from_records(zip(list_files,list_files_names), columns = ['path_file', 'name_file'])
    return df_imgs_paths

def searching_all_files(directory):
    dirpath = Path(directory)
    assert(dirpath.is_dir())
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x) # convert posix to string earlier
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list

