import torch

def train_network(
    NN=None,
    optimizer=None,
    loss_function=None,
    max_epochs=20,
    train_DataLoader=None,
    test_DataLoader=None,
    prediction_style=None
):

    # if prediction_style=='regression':
    total_loss_list_train=list()
    total_loss_list_test=list()
    # elif prediction_style=='classify':
    #     total_accuracy_list_train=list()
    #     total_accuracy_list_test=list()        


    for epoch in range(max_epochs):

        this_train_loss=0
        this_test_loss=0
        NN.train()

        for i,(X,y) in enumerate(train_DataLoader):

            if i%100==0:
                print(f'we are on batch {i}')

            yHat=NN(X)
            loss=loss_function(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('yHat')
            # print(yHat.size())
            # print('y')
            # print(y.size())
            # print('loss')
            # print(loss.size())
            # print(loss)

            if prediction_style=='classify':
                #print('statement from udemy')
                #print(100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item())
                this_train_loss+=100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item()
                # hold=input('hold')
            elif prediction_style=='regression':
                # hjold=input('hodl')
                this_train_loss+=loss.item()

        this_train_loss=this_train_loss/i
        total_loss_list_train.append(this_train_loss)


        NN.eval()
        for i,(X,y) in enumerate(test_DataLoader):
            yHat=NN(X)
            if prediction_style=='classify':
                #print('statement from udemy')
                #print(100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item())
                this_test_loss+=100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item()
                #hold=input('hold')
            elif prediction_style=='regression':
                #hjold=input('hodl')
                this_test_loss+=loss.item()
            
        this_test_loss=this_test_loss/i
        total_loss_list_test.append(this_test_loss)

    return total_loss_list_train,total_loss_list_test