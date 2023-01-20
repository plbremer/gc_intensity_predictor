import numpy as np

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
    # prediciton_style='class', 
    # include_fingerprint=True,
    # include_mz_location=True,
    # include_mz_surroundings=True,
    # learning_rate=0.001

#place to save files

#training params?

from gcmsdataset import *
from gcmsnn import *
from gcms_dataloaders import *

total_parameter_dict={
    'include_mz_location':[True],
    'include_mz_surroundings':[True],
    'include_fingerprint':[True],
    'maximum_largest_intensity':[500],
    'subsample_with_class_imbalance':[True],
    'form_as_classification':[True],
    'depth':[2],#depth is number of layers between input and output
    'num_dropout_layers':[1],#must be less than or equal to depth
    'dropout_prob':[0],
    'prediciton_style':['class'], 
    # include_fingerprint=True],
    # include_mz_location=True],
    # include_mz_surroundings=True],
    'learning_rate':[0.001],
    'batch_size':[32],
    'max_epochs':[20],
    'DataLoader_shuffle':[False],
    'num_workers':[2]
}

# my_GCMSDataset=GCMSDataset(
#     include_mz_location=True,
#     include_mz_surroundings=True,
#     include_fingerprint=True,
#     maximum_largest_intensity=500,
#     subsample_with_class_imbalance=True,
#     form_as_classification=True
# )

my_GCMSNN,my_loss_function,my_optimizer=create_GCMSNN_and_peripherals(
    depth=2,#depth is number of layers between input and output
    num_dropout_layers=1,#must be less than or equal to depth
    dropout_prob=0,
    prediciton_style='regression', 
    include_fingerprint=True,
    include_mz_location=True,
    include_mz_surroundings=True,
    learning_rate=0.001
)

junk=np.random.rand(3,3900)
junk=torch.tensor(junk).float()

yHat=my_GCMSNN(junk)
print(yHat)

train_Dataloader,test_Dataloader=
