import torch
import torch.nn as nn
from math import floor
import torch.nn.functional as F


def create_GCMSNN_and_peripherals(
    depth=2,#depth is number of layers between input and output
    #breadth='2x',
    num_dropout_layers=1,#must be less than or equal to depth
    dropout_prob=0,
    prediction_style='class', 
    #should ask dataset to provide a value in the superwrapper
    #for now we basically assume that the number of features is only a function
    #of what types of features we include (abs position, difference, structure)
    include_fingerprint=True,
    include_mz_location=True,
    include_mz_surroundings=True,
    learning_rate=0.001
):
    class GCMSNN(nn.Module):

        def __init__(
            self,
            depth=2,#depth is number of layers between input and output
            #breadth='2x',
            num_dropout_layers=1,#must be less than or equal to depth
            dropout_prob=0,
            prediction_style='class', 
            #should ask dataset to provide a value in the superwrapper
            #for now we basically assume that the number of features is only a function
            #of what types of features we include (abs position, difference, structure)
            include_fingerprint=True,
            include_mz_location=True,
            include_mz_surroundings=True
        ):

            self.num_dropout_layers=num_dropout_layers#must be less than or equal to depth
            self.dropout_prob=dropout_prob
            self.depth=depth
            
            super().__init__()
            # self.depth=depth
            # self.breadth=breadth
            # self.num_dropout_layers=num_dropout_layers
            # self.dropout_prob=dropout_prob
            # self.prediction_style=prediction_style

            #in conjunction with the comment in the input, we include these parameters as fixed assumptions
            list_of_included_feature_lengths=list()
            if include_mz_location==True:
                list_of_included_feature_lengths.append(500)
            if include_fingerprint==True:
                list_of_included_feature_lengths.append(2400)
            if include_mz_surroundings==True:
                list_of_included_feature_lengths.append(2*500)
            # length_of_morgan_fingerprints=2000
            # number_of_mz_abs_locations=500
            # number_of_relative_locations=2*500
            # total_input_feature_size=sum([length_of_morgan_fingerprints,number_of_mz_abs_locations,number_of_relative_locations])
            total_input_feature_size=sum(list_of_included_feature_lengths)

            #not done yet with prediction style, later need to determine the loss type
            if prediction_style=='class':
                total_output_size=10
            elif prediction_style=='regression':
                total_output_size=1

            #now we are free to decide the general shape of the network. Basically, we will try simple geometries for now.
            #if it seems like this works(ish) we might try tmore geomeotries
            #so we hide the breadth parameter for now and basically assume a pyramid shape going from total_input_feature_size to total_output
            #so the slope is basically (input-output)/depth
            slope=(total_input_feature_size-total_output_size)/(depth)

            #basically, the approach is to iterate through 

            self.layers=nn.ModuleDict()

            #self.layers['input']=nn.Linear(total_input_feature_size)
            #self.layers['output']=nn.Linear(total_input_feature_size)

            for i in range(depth):
                #print(self.layers)
                #there is weird remainder stuff.
                if i==0:
                    # print(total_input_feature_size)
                    # print(floor(total_input_feature_size-((i+1)*slope)))

                    self.layers[f'hidden_{i}']=nn.Linear(
                        total_input_feature_size,
                        floor(total_input_feature_size-slope)
                    )
                elif i==(depth-1):
                    #print(vars(self.layers[f'hidden_{i-1}']))
                    self.layers[f'hidden_{i}']=nn.Linear(
                        self.layers[f'hidden_{i-1}'].out_features,#['out_features'],
                        total_output_size
                    )
                else:
                    self.layers[f'hidden_{i}']=nn.Linear(
                        self.layers[f'hidden_{i-1}'].out_features,
                        floor(self.layers[f'hidden_{i-1}'].out_features-slope),
                    )
            print(self.layers)

        def forward(self, x):

            for i in range(self.depth):
                x=F.relu( self.layers[f'hidden_{i}'](x) )
                if i<self.num_dropout_layers:
                    x=F.dropout(
                        x,
                        p=self.dropout_prob,
                        training=self.training
                    )

            return x


    my_GCMSNN=GCMSNN(
        depth,#depth is number of layers between input and output
        #breadth='2x',
        num_dropout_layers,#must be less than or equal to depth
        dropout_prob,
        prediction_style, 
        #should ask dataset to provide a value in the superwrapper
        #for now we basically assume that the number of features is only a function
        #of what types of features we include (abs position, difference, structure)
        include_fingerprint,
        include_mz_location,
        include_mz_surroundings
    )


    if prediction_style=='class':
        loss_function=nn.CrossEntropyLoss()
    elif prediction_style=='regression':
        loss_function=nn.MSELoss()

    optimizer=torch.optim.Adam(my_GCMSNN.parameters(),lr=learning_rate)

    return my_GCMSNN,loss_function,optimizer

#    def generate_nn(self):
if __name__=="__main__":


    my_GCMSNN=create_GCMSNN_and_peripherals(
        depth=5,#depth is number of layers between input and output
        #breadth='2x',
        num_dropout_layers=0,#must be less than or equal to depth
        dropout_prob=0,
        prediction_style='regression', 
        #should ask dataset to provide a value in the superwrapper
        #for now we basically assume that the number of features is only a function
        #of what types of features we include (abs position, difference, structure)
        include_fingerprint=True,
        include_mz_location=True,
        include_mz_surroundings=True,
        learning_rate=0.001
    )