from torch.utils.data.dataset import Dataset
from torchvision import transforms
from src.one_time_utilities.cropper import img_downsize,img_crop_center
import pandas as pd
import numpy as np
import cv2


# transformations = transforms.Compose([transforms.Lambda(lambda x: crop_center(x)),
#                                       transforms.Lambda(lambda x: downsize(x)),
#                                       transforms.ToTensor()
#                                       ])

transformations = transforms.Compose([transforms.ToTensor()])

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path_or_df):
        """
        Args:
            csv_path_or_df (string): path to csv file or ready df
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_transform = transformations
        # Read the csv file
        if isinstance(csv_path_or_df,str):
            self.data_info = pd.read_csv(csv_path_or_df, header=None)
        else:
            self.data_info = csv_path_or_df
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column contain files names
        self.image_nam = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for label
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # Additional column can be added if needed
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = cv2.imread(single_image_name._str)

        # Check if there is an operation
        # some_operation = self.operation_arr[index]
        # If there is an operation
        # if some_operation:
            # Do some operation on image
            # ...
            # ...
            # pass
        # Transform image to tensor
        img_as_tensor = self.to_transform(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Call dataset
    custom_mnist_from_images =  \
        CustomDatasetFromImages('../data/mnist_labels.csv')