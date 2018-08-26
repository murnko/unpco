import cv2
import pandas as pd
from matplotlib import pyplot as plt
from src.utils import create_raw_data_df_list, create_csv_data
from tqdm import tqdm

"""crop&save function"""
def crop_top(df_images):
    for img_row in tqdm(df_images.itertuples()):
        path = img_row[1]
        img = cv2.imread(path._str)
        img = crop_center(img)
        img = downsize(img)
        # plt.imshow(img)
        # plt.show()

        new_path = '../../data/crop_data/'+'/'.join(path._str.split('/')[4:])
        cv2.imwrite(new_path,img)


"""lamda functions for pytorch"""
def crop_center(img):
    return img[100:400,280:560]
def downsize(img):
    return cv2.resize(img, (32,32))


if __name__ == '__main__':
    path_bi_data = '../../data/raw_data'
    df_bi_labeled = create_csv_data(path_bi_data, [2])
    crop_top(df_bi_labeled)
