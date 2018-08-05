from src.utils import create_raw_data_df_list

path_good_data = 'data/raw_data/good'
path_bad_data = 'data/raw_data/bad'

if __name__ == "__main__":
    df_good = create_raw_data_df_list(path_good_data)
    df_bad = create_raw_data_df_list(path_bad_data)