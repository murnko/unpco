from src.utils import create_raw_data_df_list, create_csv_data
from src.customdataset import CustomDatasetFromImages
import torch


path_bi_data = 'data/raw_data'

if __name__ == "__main__":
    df_bi_labeled = create_csv_data(path_bi_data, [2])
    # print(df_bi_labeled.depth_2.value_counts())
    ds_herceptin = CustomDatasetFromImages(df_bi_labeled)

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=ds_herceptin,
                                                    batch_size=10,
                                                    shuffle=True)

    for images, labels in mn_dataset_loader:
        print(images, labels)
    # print(ds_perjeta.__getitem__(0))
    # print(ds_perjeta.__len__())