import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#gcmsdataset parameters
    # my_Dataset=GCMSDataset(
    #     include_mz_location=True,
    #     include_mz_surroundings=True,
    #     include_fingerprint=True,
    #     maximum_largest_intensity=500,
    #     subsample_with_class_imbalance=True,
    #     form_as_classification=True
    # )

#gcmsnn and peripheral parameters
    # depth=2,#depth is number of layers between input and output
    # #breadth='2x',
    # num_dropout_layers=1,#must be less than or equal to depth
    # dropout_prob=0,
    # prediciton_style='classify', 
    # include_fingerprint=True,
    # include_mz_location=True,
    # include_mz_surroundings=True,
    # learning_rate=0.001

#place to save files

#training params?

from gcmsdataset import *
from gcmsnn import *
from gcms_dataloaders import *
from train_gcmsnn import *

#location of spectra np
# spectra_file_address='../intermediates/spectra_as_np_nist17gc.bin'
# fingerprint_file_address='../intermediates/gc_with_morgan.bin'
#location of fingerprint file
#fingerprint file is a subset of the spectra file
#not every structure was reliable, and there were also multiple isomer forms of various flattened
#we arbitrarily kept the first encountering of each inchikey first block
#we also only made fingerprints for those compounds with an observed max mz <=500 and >50 
spectra_file_address='../intermediates/mini_spectra_as_np_nist17gc.bin'
fingerprint_file_address='../intermediates/mini_gc_with_morgan.bin'
spectra_data=pd.read_pickle(spectra_file_address)
structure_data=pd.read_pickle(fingerprint_file_address)



def custom_collate(data):
    '''
    receives a list of tuples and returns a tuple (remember we receive x,y)
    '''
    # print('this is incoming data')
    # print(data)
    # print(len(data))
    return torch.cat([element[0] for element in data]),torch.cat([element[1] for element in data])
    #return data


total_parameter_dict={
    'include_mz_location':[True],
    'include_mz_surroundings':[True],
    'include_fingerprint':[True,False],
    'maximum_largest_intensity':[501],
    'subsample_with_class_imbalance':[True],
    'depth':[2,4],#depth is number of layers between input and output
    'num_dropout_layers':[0,1],#must be less than or equal to depth
    'dropout_prob':[0.3],
    'prediction_style':['classify','regression'], 
    'learning_rate':[0.0001,0.001,0.01,0.1],
    'batch_size':[32],
    'max_epochs':[20],
    'shuffle':[True],
    'num_workers':[2],
    'structure_data':[structure_data],
    'spectra_data':[spectra_data]
}

my_GCMSDataset=GCMSDataset(
    include_mz_location=True,
    include_mz_surroundings=True,
    include_fingerprint=True,
    maximum_largest_intensity=501,
    subsample_with_class_imbalance=True,
    prediction_style='classify',
    structure_data=structure_data,
    spectra_data=spectra_data
)

my_GCMSNN,my_loss_function,my_optimizer=create_GCMSNN_and_peripherals(
    depth=2,#depth is number of layers between input and output
    num_dropout_layers=1,#must be less than or equal to depth
    dropout_prob=0,
    prediction_style='classify', 
    include_fingerprint=True,
    include_mz_location=True,
    include_mz_surroundings=True,
    learning_rate=0.001
)

train_Dataloader,test_Dataloader=create_dataloaders(
    my_GCMSDataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    collate_function=custom_collate
)
train_loss,test_loss=train_network(
    NN=my_GCMSNN,
    optimizer=my_optimizer,
    loss_function=my_loss_function,
    max_epochs=2,
    train_DataLoader=train_Dataloader,
    test_DataLoader=test_Dataloader,
    prediction_style='classify'
)
plt.scatter(
    range(len(train_loss)),
    train_loss
)
plt.show()
#plt.savefig('../results/regression_train.png')
plt.scatter(
    range(len(test_loss)),
    test_loss
)
#plt.savefig('../results/regression_test.png')
plt.show()





