import cv2
import pandas as pd
from matplotlib import pyplot as plt
from src.utils import create_raw_data_df_list, create_csv_data
from tqdm import tqdm
import multiprocessing as mp

t_new_ref = cv2.imread('../../data/aux_data/double_corner.png')
t_new_ref_gray = cv2.cvtColor(t_new_ref, cv2.COLOR_BGR2GRAY)

"""crop&save function"""
def crop_top(df_images):
    for img_row in tqdm(df_images.itertuples()):
        path = img_row[1]
        img = cv2.imread(path._str)
        img = img_crop_center(img)
        img = img_downsize(img)
        # plt.imshow(img)
        # plt.show()

        new_path = '../../data/crop_data_0/'+'/'.join(path._str.split('/')[4:])
        cv2.imwrite(new_path,img)

def crop_ref(img_row):
    # print(img_row)
    path = img_row[1]
    try:
        img = cv2.imread(path)
        img_crop = img_crop_ref(img)
        img_crop32 = img_downsize(img_crop,32)
        img_crop64 = img_downsize(img_crop,64)
        img_crop128 = img_downsize(img_crop,128)
        new_path = '../../data/crop_data_32/' + path.split('/')[-1]
        cv2.imwrite(new_path, img_crop32)
        new_path = '../../data/crop_data_64/' + path.split('/')[-1]
        cv2.imwrite(new_path, img_crop64)
        new_path = '../../data/crop_data_128/' + path.split('/')[-1]
        cv2.imwrite(new_path, img_crop128)
        return 0
    except AttributeError:
        print(img_row)
        return -1



"""lambda functions for pytorch"""


def img_crop_center(img):
    return img[100:400,280:560]


def img_downsize(img,n):
    return cv2.resize(img, (n,n))


def img_crop_ref(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, t_new_ref_gray, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    return img[:, top_left[0]:]



if __name__ == '__main__':
    path_bi_data = '/home/murnko/PycharmProjects/unpco/data/pCO2'
    df_bi_labeled = create_raw_data_df_list(path_bi_data)
    df_bi_labeled['path_file'] = df_bi_labeled['path_file'].apply(lambda x: x._str)
    # print(df_bi_labeled['name_file'].value_counts())
    df_list = [[x[0], x[1], x[2]] for x in df_bi_labeled.itertuples()]
    pool = mp.Pool(processes=10)
    results = [pool.apply_async(crop_ref, args=(img_row,)) for img_row in df_list]
    results = [p.get() for p in results]
    results.sort()
    print(max(results))
    print(min(results))
