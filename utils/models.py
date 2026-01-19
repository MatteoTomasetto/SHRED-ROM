import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from .processdata import mse, mre, num2p

class SHRED(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, decoder_sizes = [350, 400], dropout = 0.0):
        '''
        SHRED model definition
        
        Inputs
        	input size (e.g. number of sensors)
        	output size (e.g. full-order variable dimension)
        	size of LSTM hidden layers (default to 64)
        	number of LSTM hidden layers (default to 2)
        	list of decoder layers sizes (default to [350, 400])
        	dropout parameter (default to 0)
        '''
            
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers,
                                  batch_first=True)
        
        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, hidden_size)
        decoder_sizes.append(output_size)

        for i in range(len(decoder_sizes)-1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i != len(decoder_sizes)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (output, _) = self.lstm(x, (h_0, c_0))
        output = output[-1].view(-1, self.hidden_size)

        for layer in self.decoder:
            output = layer(output)

        return output

    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

def fit(model, train_dataset, valid_dataset, batch_size = 64, epochs = 4000, optim = torch.optim.Adam, lr = 1e-3, loss_fun = mse, loss_output = mre, formatter = num2p, verbose = False, patience = 5):
    '''
    Neural networks training
    
    Inputs
    	model (`torch.nn.Module`)
    	training dataset (`torch.Tensor`)
    	validation dataset (`torch.Tensor`)
    	batch size (default to 64)
    	number of epochs (default to 4000)
    	optimizer (default to `torch.optim.Adam`)
    	learning rate (default to 0.001)
        loss function (defalut to Mean Squared Error)
        loss value to print and return (default to Mean Relative Error)
        loss formatter for printing (default to percentage format)
    	verbose parameter (default to False) 
    	patience parameter (default to 5)
    '''

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    optimizer = optim(model.parameters(), lr = lr)

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            def closure():
                outputs = model(data[0])
                optimizer.zero_grad()
                loss = loss_fun(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + formatter(train_error_list[-1]) + " \t Validation loss = " + formatter(valid_error_list[-1]) + " "*10, end = "\r")

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            
            if verbose == True:
                print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
         
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    model.load_state_dict(best_params)
    train_error = loss_output(train_dataset.Y, model(train_dataset.X))
    valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
    
    if verbose == True:
    	print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
    
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
 
def forecast(forecaster, input_data, steps, nsensors):
    '''
    Forecast time series in time
    Inputs
    	forecaster model (`torch.nn.Module`)
        starting time series of dimension (ntrajectories, lag, nsensors+nparams)
    	number of forecasting steps
        number of sensors
    Outputs
        forecast of the time series in time
    '''   

    forecast = []
    for i in range(steps):
        forecast.append(forecaster(input_data))
        temp = input_data.clone()
        input_data[:,:-1] = temp[:,1:]
        input_data[:,-1, :nsensors] = forecast[i]

    return torch.stack(forecast, 1)














