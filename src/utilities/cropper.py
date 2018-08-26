import cv2
import pandas as pd
from matplotlib import pyplot as plt

def crop_top(df_images):
    for img_row in df_images.itertuples():
        path = img_row[1]
        img = cv2.imread(path._str)
        img = crop_center(img)
        img = downsize(img)
        plt.imshow(img)
        plt.show()
        # if plt.waitforbuttonpress():
        #     continue

def crop_center(img):
    return img[100:400,280:560]
def downsize(img):
    return cv2.resize(img, (32,32))