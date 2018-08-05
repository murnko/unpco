from src.utils import create_raw_data_df_list
from src.customdataset import CustomDatasetFromImages
import torch


path_bi_data = 'data/raw_data'

if __name__ == "__main__":
    df_bi_labeled = create_raw_data_df_list(path_bi_data)
    print(df_bi_labeled.head())
    ds_perjeta = CustomDatasetFromImages(df_bi_labeled)

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=ds_perjeta,
                                                    batch_size=10,
                                                    shuffle=True)

    for images, labels in mn_dataset_loader:
        pass
    # print(ds_perjeta.__getitem__(0))
    # print(ds_perjeta.__len__())