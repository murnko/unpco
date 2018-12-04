import pandas as pd
from pathlib import Path

def create_raw_data_df_list(dir_path):
    list_files = searching_all_files(dir_path)
    list_files_names = [x._str.split('/')[-1] for x in list_files]
    df_imgs_paths = pd.DataFrame.from_records(zip(list_files,list_files_names), columns = ['path_file', 'name_file'])
    df_imgs_paths['label'] = df_imgs_paths['path_file'].apply(lambda x: x._str.split('05_F')[-1][:2])
    return df_imgs_paths

def searching_all_files(directory):
    dirpath = Path(directory)
    assert(dirpath.is_dir())
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file() and '.bmp' in x._str:
            file_list.append(x) # do not convert posix to string earlier
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list


def create_csv_data(dir_path, lvl2col = []):
    list_files = searching_all_files(dir_path)
    list_files_names = [x._str.split('/')[-1] for x in list_files]
    df_imgs_paths = pd.DataFrame.from_records(zip(list_files,list_files_names), columns = ['path_file', 'name_file'])
    for k in lvl2col:
        df_imgs_paths['depth_%s' % str(k)] = df_imgs_paths['path_file'].apply(lambda x: x._str.split('/')[int(k)])
    return df_imgs_paths
