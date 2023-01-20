from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from random import random
import matplotlib.pyplot as plt



class GCMSDataset(Dataset):
    
    def coerce_one_spectrum_to_difference_features_and_output_intensity(self,temp_spectrum):
        '''
        this has parameters about what to include but we assignthem in the intilization of the class
        '''

        difference_features=list()
        total_intensities=list()
        #should be refactored for vectorization
        for i in range(len(temp_spectrum)):
            this_peaks_differences=list()
            for j in range(len(temp_spectrum)):
                if i==j:
                    continue
                this_peaks_differences.append(temp_spectrum[j,0]-temp_spectrum[i,0])
            difference_features.append(this_peaks_differences)
            total_intensities.append(temp_spectrum[i,1])
  
        values_to_set_at_one=np.array(difference_features,dtype=int)
        
        values_to_set_at_one=values_to_set_at_one+self.maximum_largest_intensity
        values_to_set_at_one
        x=[np.zeros(2*self.maximum_largest_intensity,dtype=np.float32) for counter in range(len(values_to_set_at_one))]    
        x=np.vstack(x)

        for counter in range(len(values_to_set_at_one)):
            x[counter,values_to_set_at_one[counter]]=1

        y=np.array(total_intensities)

        return x,y

    def coerce_one_spectrum_to_mz_position(self,temp_spectrum):

        output_array=np.zeros(
            (len(temp_spectrum),self.maximum_largest_intensity)
        )

        #set the particular mz to 1
        #we set one value in each row to one
        #the mz are the things that we set to one
        output_array[
            [range(len(temp_spectrum))], temp_spectrum[:,0].astype(int)
        ]=1

        return output_array

    def return_fingprint_repeated_num_of_mz_times(self, temp_fingerprint, temp_spectrum):

        fingerprint_list=[temp_fingerprint for i in range(len(temp_spectrum))]

        return np.vstack(fingerprint_list)

    def __init__(
        self,
        include_mz_location=True,
        include_mz_surroundings=True,
        include_fingerprint=True,
        maximum_largest_intensity=500,
        subsample_with_class_imbalance=True,
        prediction_style='class',
        spectra_data=None,
        structure_data=None
    ):



        self.spectra_data=spectra_data
        self.structure_data=structure_data        

        #in light of the above notes on the fingerprints, we subset the spectra
        self.spectra_data=self.spectra_data.loc[self.structure_data.index,:]

        self.spectra_data.reset_index(inplace=True,drop=True)
        self.structure_data.reset_index(inplace=True,drop=True)

        self.include_mz_location=include_mz_location
        self.include_mz_surroundings=include_mz_surroundings
        self.include_fingerprint=include_fingerprint
        self.maximum_largest_intensity=maximum_largest_intensity
        self.subsample_with_class_imbalance=subsample_with_class_imbalance
        self.prediction_style=prediction_style

    def __len__(self):
        return len(self.spectra_data.index)
        
    def __getitem__(self,idx):
        #getting an item is less straightforward than most things
        #we are training on individual mz, so we would like to have a file that is simply those
        #but it would take up too much memory
        #instead, what we do is basically request n spectra
        #and for each spectra, produce a large number of mz
        #so we should basically make a function that gets one, concatenate n of them, return that
        
        
        #we want to get
        #the absolute mz placement
        #the difference compared to surrounding mz
        #should decouple the output and difference features, but its ok
        
        included_feature_list=list()

        if self.include_mz_surroundings==True:
            spectrum_differences,y=self.coerce_one_spectrum_to_difference_features_and_output_intensity(
                self.spectra_data.at[idx,'spectrum_np']
            )
            included_feature_list.append(spectrum_differences)

        if self.include_mz_location==True:
            included_feature_list.append(self.coerce_one_spectrum_to_mz_position(
                self.spectra_data.at[idx,'spectrum_np']
            ))
            

        if self.include_fingerprint==True:
            included_feature_list.append(self.return_fingprint_repeated_num_of_mz_times(
                self.structure_data.at[idx,'morgan_fingerprints'],
                self.spectra_data.at[idx,'spectrum_np']
            ))

        if self.include_mz_surroundings==True and self.include_mz_location==True and self.include_fingerprint==True:
            x=np.hstack(included_feature_list)

        #y=

        if self.subsample_with_class_imbalance==True:
            #give each value a proability of getting kept
            #compare to likelihood to keep "cutoffs"
            #manually obtained by looking at distro in other notebook
            #eg, class 9 is the rarest and is always kept
            cutoffs=[
                0.006009713289612066,
                0.035906286754599816,
                0.09002860848854079,
                0.1648890599585294,
                0.27565783556013757,
                0.4087369576592012,
                0.563356382301786,
                0.7710022642385772,
                1.0,
                0.1515232191077473
            ]
            
            # print('---------')
            # print(y)
            classes=np.digitize(y,bins=np.arange(0,1,0.1))
            classes=[element-1 for element in classes]
            # print(classes)
            random_numbers=np.random.rand(len(y))
            cutoff_per_element=[cutoffs[class_type] for class_type in classes]
            #to show it works we just need to show that the hist before is about what we saw across all
            #andthe hist after is about equal
            # plt.hist(y,bins=10)
            # plt.show()
            y=y[
                random_numbers<cutoff_per_element
            ]
            # plt.hist(y,bins=10)
            # plt.show()
            x=x[
                random_numbers<cutoff_per_element
            ]


        if self.prediction_style=='class':
            y=np.digitize(y,bins=np.arange(0,1,0.1))
            y=np.array([element-1 for element in y])
        #also, we want to subsample if that is true


        # print(np.shape(x))
        # #print(x)
        # print(np.shape(y))
        # #print(y)

        #what is the difference here? from vs tensor
        if self.prediction_style=='regression':
            x=torch.tensor(x).float()
            y=torch.from_numpy(y).float()
        elif self.prediction_style=='class':
            x=torch.tensor(x).float()
            y=torch.from_numpy(y).long()
        

        return x,y

def custom_collate(data):
    '''
    receives a list of tuples and returns a tuple (remember we receive x,y)
    '''
    # print('this is incoming data')
    # print(data)
    # print(len(data))
    return torch.cat([element[0] for element in data]),torch.cat([element[1] for element in data])
    #return data

# if __name__=="__main__":
    #notes to self: what we really want to check out is
    #include fingerprints: yes or no
    #frame as classification: yes or no
    #include absolute position: yes or no (probably yes?)

    #my_Dataset.__getitem__(0)

    # test_dataloader=DataLoader(
    #     dataset=my_Dataset,
    #     #should keep the bin size high to guarantee never rolling "keeping 0" n times
    #     batch_size=50,
    #     shuffle=False,
    #     num_workers=1,
    #     collate_fn=custom_collate
    # )

    # test_data=iter(test_dataloader)
    # X,y=next(test_data)
    # print(X)
    # print(y)
    # plt.hist(y.numpy(),bins=10)
    # plt.show()
