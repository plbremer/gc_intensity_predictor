from torch.utils.data import Dataset, DataLoader,random_split

def create_dataloaders(
    my_dataset,
    batch_size,
    shuffle,
    num_workers
):

    train_set,test_set=random_split(my_dataset,lengths=(0.8,0.2))

    train_DataLoader=DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    test_DataLoader=DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    return train_DataLoader,test_DataLoader
