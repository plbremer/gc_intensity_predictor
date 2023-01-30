from torch.utils.data import Dataset, DataLoader,random_split




def create_dataloaders(
    my_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    collate_function=None,
    train_test_split=0.8
):

    if train_test_split!=0:
        train_set,test_set=random_split(my_dataset,lengths=(train_test_split,1-train_test_split))
    elif train_test_split==0:
        train_set=None
        test_set=my_dataset

    train_DataLoader=DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_function
    )

    test_DataLoader=DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_function
    )

    return train_DataLoader,test_DataLoader
