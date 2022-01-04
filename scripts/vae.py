#!/usr/bin/env python


# Basic libraries
import os
import numpy as np
import pickle
import pandas as pd
import re
import matplotlib.pyplot as plt
import argparse

# Torch dependencies
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F


# Custom methods
from pytorch_loading import SNPLoading
from preprocess_vae import one_hot_encoding, metadata_mapping, split_train_test, impute_data, get_enc_dict, loss_ignore_nans
from vae_out import de_encoding


from mpl_toolkits.mplot3d import Axes3D

# Import Tensorboard
from torch.utils.tensorboard import SummaryWriter

##################################################
### Initialize hyper parameters, CUDA and seed ###
##################################################

parser = argparse.ArgumentParser(description='VAE ancestry project for DL course, by Leonardo Cobuccio')

parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 30)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status (default: 1)')

parser.add_argument("--hidden-layers",default=3,type=int,
                    help='number of hidden layers, default=3.')

parser.add_argument("--hidden-units",default=100,type=int,
                    help='units, nodes per hidden layer. default=100')  

parser.add_argument("--prefix",default="",type=str,
                    help='Prefix to add to the results folder name')  
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


CUDA = torch.cuda.is_available()
SEED = args.seed
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
HIDDEN_UNITS = args.hidden_units
HIDDEN_LAYERS = args.hidden_layers
PREFIX = args.prefix

# Fixed hyperparameters
ZDIMS = 2 #Replace with your value (Dimensions of latent space)
TRAIN = 0.8 #Replace with your value (proportion of samples to keep on training set)

# Print out hyperparameters chosen
print("batch size chosen:", BATCH_SIZE)
print("epochs chosen:", EPOCHS)
print("hidden layers chosen:", HIDDEN_LAYERS)
print("hidden units chosen:", HIDDEN_UNITS)

# Hyperparameters to play with at the beginning of writing the script
# CUDA = torch.cuda.is_available()
# SEED = 42 #Replace with your value
# BATCH_SIZE = 20 #Replace with your value 
# EPOCHS = 5 #Replace with your value
# ZDIMS = 2 #Replace with your value (Dimensions of latent space)
# TRAIN = 0.8 #Replace with your value (proportion of samples to keep on training set)
# HIDDEN_UNITS = 100 #Replace with your value (Units per layer)
# HIDDEN_LAYERS = 2 #Replace with your value (Amount of hidden layers)


# I save the date and the hour
now = datetime.now()
date_string = now.strftime("%d_%m_%Y_%H_%M")

# I save the experiment ID with the hyperparameters set and the date at the end
experiment_id = f"{PREFIX}hunits{HIDDEN_UNITS}_hlayers{HIDDEN_LAYERS}_epochs{EPOCHS}_batchsize{BATCH_SIZE}_zdims{ZDIMS}_trainprop{TRAIN}_{date_string}"

# Create experiment directory
experiment_directory = "../results/" + experiment_id
os.makedirs(experiment_directory)

# Creates a folder in which to store the Tensorboard metrics
writer = SummaryWriter("runs/" + experiment_id)

# Set seed to GPU
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# Allow use of GPU memory
device = torch.device("cuda" if CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

#####################################
### Map metadata to observations  ###
#####################################

# Get files path
data_files_path = "../tensor_data/"
files = os.listdir(data_files_path)
files.sort()

# Map metadata to sample encodings/name to get labels
encodings_file = data_files_path+files[0]
metadata_path = "../metadata/v44_metadata_clear.tsv"

# Remove encoding file and variants file
X = files[1:-1] 
features = files[-1]

##############################################################################
### Solve problem of unsorted patients and of incorrectly mapped Xs and Ys ###
##############################################################################

# I remove the "_patient.pt" at the end of each patient string and I convert it to integer
stripped_integers = [int(i.rstrip("_patient.pt")) for i in X]
# I sort it numerically, so from 1 to 6155
stripped_integers.sort()

# I reconvert the patients number in string and concatenate with "_patient.pt"
X = [str(i) + "_patient.pt" for i in stripped_integers]

# Contains the 6155 targets, i.e. the ancestries ['Africa', 'North Asia', 'South East Europe'] etc
targets = metadata_mapping(X, metadata_path)

#########################################
### Encode targets : One-hot-encoding ###
#########################################

original_targets = np.array(targets)
targets = one_hot_encoding(targets)

# Make encoding dict to map encoding to original target
dict_encoding = get_enc_dict(original_targets, targets)
with open('../results/enc_dict', 'wb') as handle:
    pickle.dump(dict_encoding, handle)

####################################
### Partition train and test set ###
####################################

X_train, X_test, y_train, y_test = split_train_test(X, targets, 0.8)

train_set = SNPLoading(data_path=data_files_path, data_files=X_train, targets=y_train)
test_set = SNPLoading(data_path=data_files_path, data_files=X_test, targets=y_test)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)



##################
### VAE Module ###
##################

# Get input features and encoding len of targets
## Explanation:
## train_set[0] takes the first element of the training set, consisting of a tuple of two tensors: the X and the Y
## train_set[0][0] selects the X, i.e. the SNPs encoded as [0,0], [0,1], or [1,1]
## train_set[0][0].shape[0] takes the number of features, i.e. of SNPs (109348)
input_features = train_set[0][0].shape[0]
target_enc_len = targets.shape[1]


class VAE(nn.Module):
    def __init__(self, input_features, input_batch, zdims,hidden_units, hidden_layers):
        super(VAE, self).__init__()
        
        # Input data
        self.input_features = input_features
        self.input_batch = input_batch
        self.zdims = zdims
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.relu = nn.ReLU()
        
        ### ENCODER : From input dimension to bottleneck (zdims)
        ## Input layer (fc1 : fully connected layer 1)

        self.fc0 = nn.Linear(in_features = 2,
                                out_features = 1)
        self.bn0 = nn.BatchNorm1d(num_features = self.input_features)

        #   Implement your code 
        self.fc1 = nn.Linear(in_features = self.input_features,
                                out_features = self.hidden_units)
        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_units)


        # Hidden layers (fcn)
        self.encode_layers = nn.ModuleList()
        self.encode_batchnorms = nn.ModuleList()

        for i in range(0, hidden_layers):
            self.encode_layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            self.encode_batchnorms.append(nn.BatchNorm1d(self.hidden_units))

        ## Hidden to latent (fc3.1, fc3.2)
        #   Implement your code
        self.fcmu = nn.Linear(in_features = self.hidden_units,
                                out_features = self.zdims)
        self.bnmu = nn.BatchNorm1d(num_features = self.zdims)

        self.fclogvar = nn.Linear(in_features = self.hidden_units,
                                    out_features = self.zdims)
        self.bnlogvar = nn.BatchNorm1d(num_features = self.zdims)


        ### DECODER : From bottleneck to input dimension
        ## Latent to first hidden (fc4)
        #   Implement your code
        self.fc4 = nn.Linear(in_features = self.zdims,
                                    out_features = self.hidden_units)
        self.bn4 = nn.BatchNorm1d(num_features = self.hidden_units)

        ## Hidden Layers (fcm)
        self.decoder_layers = nn.ModuleList()
        self.decoder_batchnorms = nn.ModuleList()

        for i in range(0, hidden_layers):
            self.decoder_layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            self.decoder_batchnorms.append(nn.BatchNorm1d(self.hidden_units))

        ## Hidden to reconstructed input (fc5)
        #   Implement your code
        self.fc4toreconstructed = nn.Linear(in_features = self.hidden_units,
                                    out_features = self.input_features)
        self.bn4toreconstructed = nn.BatchNorm1d(num_features = self.input_features)
        
        # To have the One-hot encoding of the genotypes again 
        self.fc5 = nn.Linear(1,2)

    def encode(self, x, impute=True):
        """Input vector x -> fully connected layer 1 -> ReLU -> (fc21, fc22)
        Parameters
        ----------
        x : [input_batch, input_features] matrix

        Returns
        -------
        mu     : zdims mean units one for each latent dimension (fc21(h1))
        logvar :  zdims variance units one for each latent dimension (fc22(h1))
        """
        ## Input features -> hidden_units (fc1)
        #   Implement your code
        # print("Input as it is")
        # print("x.shape -->", x.shape) # torch.Size([30, 109348, 2])
        # First fully connected layer to encode the one-hot encoding as 1 single value
        # So the shape changes from # torch.Size([30, 109348, 2]) to # torch.Size([30, 109348, 1])
        h1 = self.relu(self.bn0(self.fc0(x))) # torch.Size([30, 109348, 1])
        # print("First fully connected layer to encode one-hot encoding")
        # print("h1.shape -->", h1.shape)
        # print(h1[0:10])
        # I need to reshape the torch.Size([30, 109348, 1]) to torch.Size([30, 109348]), because that's how the linear layer wants the dimensions
        h1 = torch.reshape(h1, (h1.shape[0], self.input_features)) # torch.Size([30, 109348])
        # print(h1[0:10])
        # print("Reshaping the tensor")
        # print("h1.reshaped -->", h1.shape)
        h1 = self.relu(self.bn1(self.fc1(h1))) # torch.Size([30, 100])
        # print("fc1 layer")
        # print("h1.shape -->", h1.shape)


        ## Hidden_units -> hidden units (fcn)
        for i in range(0,self.hidden_layers):
            h1 = self.relu(self.encode_batchnorms[i](self.encode_layers[i](h1)))


        ## Hidden_units -> latent space (fc3.1,fc3.2, zdims)
        # Implement your code
        
        mu = self.bnmu(self.fcmu(h1)) # torch.Size([30, 2])
        logvar = self.bnlogvar(self.fclogvar(h1)) # torch.Size([30, 2])

        # print("mu.shape -->", mu.shape)
        # print("logvar.shape -->", logvar.shape)
        return mu, logvar

    def reparameterize(self, mu, logvar, inference=False):
        """Reparametrize to have variables instead of distribution functions
        Parameters
        ----------
        mu     : [input_batch, zdims] mean matrix
        logvar : [input_batch, zdims] variance matrix

        Returns
        -------
        During training random sample from the learned zdims-dimensional
        normal distribution; during inference its mean.
        """
        # Standard deviation
        std = torch.exp(0.5*logvar)
        # Noise term epsilon
        eps = torch.rand_like(std)
        
        if inference is True:
            return mu

        return mu+(eps*std)

    def decode(self, z):
        """z sample (20) -> fc3 -> ReLU (400) -> fc4 -> sigmoid -> reconstructed input
        Parameters
        ----------
        z : z vector

        Returns
        -------
        Reconstructed x'
        """
        ## zdims -> hidden
        #   Implement your code
        # print("z.shape -->", z.shape)
        h2 = self.relu(self.bn4(self.fc4(z))) # torch.Size([30, 100])
        # print("h2 = self.relu(self.fc4(z))")
        # print("h2.shape -->", h2.shape)
        ## Hidden -> hidden

        for i in range(0,self.hidden_layers):
            h2 = self.relu(self.decoder_batchnorms[i](self.decoder_layers[i](h2)))
        
        # Hidden -> input features
        #   Implement your code
        h2 = self.relu(self.bn4toreconstructed(self.fc4toreconstructed(h2))) # torch.Size([30, 109348])
        # print("h2 = self.relu(self.fc4toreconstructed(h2))")
        # print("h2.shape -->", h2.shape)
        h2 = torch.reshape(h2, (h2.shape[0], self.input_features, 1)) # torch.Size([30, 109348, 1])
        # print("h2 = torch.reshape(h2, (h2.shape[0], self.input_features, 1))")
        # print("h2.shape -->", h2.shape)
        h2 = self.fc5(h2) # torch.Size([30, 109348, 2])
        # print("h2 = self.fc5(h2)")
        # print("h2.shape -->", h2.shape)

        return h2

    def forward(self, x):
        """Connects encoding and decoding by doing a forward pass"""
        # Get mu and logvar
        mu, logvar = self.encode(x)
        # Get latent samples
        z = self.reparameterize(mu, logvar)
        
        # Reconstruct input
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, imputed_data, mu, logvar, input_features, input_batch, inference=False):
    """Computes the ELBO Loss (cross entropy + KLD)"""
    # KLD is Kullback–Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= (input_batch * ZDIMS)

    # Compute loss between imputed x and reconstructed imputed x
    loss = nn.BCEWithLogitsLoss(reduction="none")
    BCE = loss(recon_x, imputed_data)


    # Compute BCE and ignore values that come from a nan
    BCE = loss_ignore_nans(BCE, x) ### This function is empty, you have to implement it

    BCE = torch.sum(BCE)/len(BCE)
    
    # print(f"BCE : {BCE}, KLD : {KLD}")
    return BCE + KLD, BCE, KLD


def train(epoch, model, train_loader, CUDA, optimizer, input_features, input_batch):
    # toggle model to train mode
    model.train()
    train_loss = 0
    epoch_bce = 0
    epoch_kld = 0
    
    # Init save training
    train_loss_values = []
    train_bce = []
    train_kld = []

    # Iterate over train loader in batches of batch_size (e.g. 30)
    for batch_idx, (data, _) in enumerate(train_loader):
        
        if CUDA:
            data = data.to(device)
        
        optimizer.zero_grad()
        imputed_data = impute_data(tensor=data.cpu(), batch_size=input_batch ,categorical=True)  # Impute data function is also empty, you have to implement it

        if torch.cuda.is_available():
            data = data.to(device)
            imputed_data = imputed_data.to(device)


        # Push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(imputed_data)
        # calculate loss function
        loss, bce, kld = loss_function(recon_batch, data, imputed_data, mu, logvar, input_features, input_batch)
        
        # calculate the gradient of the loss w.r.t. the graph leaves
        loss.backward()
        train_loss += loss.detach().item()
        epoch_bce += bce.detach().item()
        epoch_kld += kld.detach().item()


        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        # Append values to then save them
        train_loss_values.append(loss.item())
        train_bce.append(bce.item())
        train_kld.append(kld.item())

        # Metrics per iteration
        writer.add_scalar("Loss_iteration/train", loss.item(), batch_idx)
        writer.add_scalar("BCE_iteration/train", bce.item(), batch_idx)
        writer.add_scalar("KLD_iteration/train", kld.item(), batch_idx)

        # # Average metrics per epoch

        # writer.add_scalar("Loss_epoch/train", loss.item(), epoch)
        # writer.add_scalar("BCE_epoch/train", bce.item(), epoch)
        # writer.add_scalar("KLD_epoch/train", kld.item(), epoch)
        print(f"Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]     BCE: {bce},     KLD: {kld}")
    # print(f"BCE : {BCE}, KLD : {KLD}")

    print('====> Epoch: {} Train Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


    train_loss_epoch = train_loss / len(train_loader.dataset)
    train_bce_epoch = epoch_bce / len(train_loader.dataset)
    train_kld_epoch = epoch_kld / len(train_loader.dataset)

    return train_loss_values, train_bce, train_kld, train_loss_epoch, train_bce_epoch, train_kld_epoch


def test(epoch, model, test_loader, CUDA, optimizer, input_features, input_batch, test_classes, zdims):
    # toggle model to test / inference mode
    test_loss = 0
    epoch_bce = 0
    epoch_kld = 0
    model.eval()

    # Save test loss
    test_loss_values = []
    test_bce = []
    test_kld = []

    # Save latent space
    mu_test = np.empty([0,zdims])
    targets_test = np.empty([0, test_classes])

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):

            #Impute data
            imputed_data = impute_data(tensor=data.cpu(), batch_size=input_batch, categorical=True)

            if CUDA:
                data = data.to(device)
                imputed_data = imputed_data.to(device)

            # Push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = model(imputed_data)
            
            # calculate loss function
            loss, bce, kld = loss_function(recon_batch, data, imputed_data, mu, logvar, input_features, input_batch)
            
            
            test_loss += loss.detach().item()
            epoch_bce += bce.detach().item()
            epoch_kld += kld.detach().item()
            print(f"Test epoch: {epoch} [{batch_idx * len(data)}/{len(test_loader.dataset)}]")
            
            # Save test error 
            test_loss_values.append(loss.item())
            test_bce.append(bce.item())
            test_kld.append(kld.item())

            writer.add_scalar("Loss_iteration/test", loss.item(), batch_idx)
            writer.add_scalar("BCE_iteration/test", bce.item(), batch_idx)
            writer.add_scalar("KLD_iteration/test", kld.item(), batch_idx)

            # Save latent space
            mu_ = mu.cpu().detach().numpy()
            target = _.cpu().detach().numpy()
            mu_test = np.append(mu_test, mu_, axis=0)
            targets_test = np.append(targets_test,target, axis=0)

        print('====> Epoch: {} Test Average loss: {:.4f}'.format(
            epoch, test_loss / len(test_loader.dataset)))


        test_loss_epoch = test_loss / len(test_loader.dataset)
        test_bce_epoch = epoch_bce / len(test_loader.dataset)
        test_kld_epoch = epoch_kld / len(test_loader.dataset)
            
        # test_loss_epoch = test_loss_epoch.item()
        # test_loss /= len(test_loader.dataset)
        

        print('====> Test set loss: {:.4f}'.format(loss))



        return test_loss_values, test_bce, test_kld, mu_test, targets_test, test_loss_epoch, test_bce_epoch, test_kld_epoch



# Call model to device
model = VAE(input_features=input_features, input_batch=BATCH_SIZE, zdims=ZDIMS, hidden_units=HIDDEN_UNITS, hidden_layers=HIDDEN_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Initialize empty lists to store Metrics per iteration
train_loss_values = []
train_bce = []
train_kld = []

test_loss_values = []
test_bce = []
test_kld = []

# Initialize empty lists to store  Metrics per epoch
train_loss_epoch_list = []
train_bce_epoch_list = []
train_kld_epoch_list = []

test_loss_epoch_list = []
test_bce_epoch_list = []
test_kld_epoch_list = []

# List of epochs at which saving the model
epochs_save = [1, 5, 10, 20, 35, 50]

# Print model architecture to txt file
model_arch = str(model)
with open(experiment_directory + "/model_arch.txt", "w") as file:
    file.write(model_arch)

# Training and testing
for epoch in range(1, EPOCHS + 1):
    train_loss_, train_bce_, train_kld_, train_loss_epoch, train_bce_epoch, train_kld_epoch  = train(epoch, model, train_loader, CUDA, optimizer, input_features, BATCH_SIZE)
    test_loss_, test_bce_, test_kld_, mu_test, targets_test, test_loss_epoch, test_bce_epoch, test_kld_epoch = test(epoch, model, test_loader, CUDA, optimizer, input_features, BATCH_SIZE, target_enc_len, ZDIMS)
    
    train_loss_values = train_loss_values + train_loss_
    train_bce = train_bce + train_bce_
    train_kld = train_kld + train_kld_

    test_loss_values = test_loss_values + test_loss_
    test_bce = test_bce + test_bce_
    test_kld = test_kld + test_kld_

    ###############################################
    #### Tensorboard: saving metrics per epoch ####
    ###############################################

    ### TRAIN metrics
    writer.add_scalar("Loss_epoch/train", train_loss_epoch, epoch)
    writer.add_scalar("BCE_epoch/train", train_bce_epoch, epoch)
    writer.add_scalar("KLD_epoch/train", train_kld_epoch, epoch)

    ### TEST metrics
    writer.add_scalar("Loss_epoch/test", test_loss_epoch, epoch)
    writer.add_scalar("BCE_epoch/test", test_bce_epoch, epoch)
    writer.add_scalar("KLD_epoch/test", test_kld_epoch, epoch)

    #############################################################
    #### To lists: aving metrics per epoch and per iteration ####
    #############################################################

    ### TRAIN metrics
    train_loss_epoch_list.append(train_loss_epoch)
    train_bce_epoch_list.append(train_bce_epoch)
    train_kld_epoch_list.append(train_kld_epoch)

    ### TEST metrics
    test_loss_epoch_list.append(test_loss_epoch)
    test_bce_epoch_list.append(test_bce_epoch)
    test_kld_epoch_list.append(test_kld_epoch)

    # Original savings as done by Felix the TA
    with open(experiment_directory + "/train_loss.pickle", "wb") as fp:
        pickle.dump(train_loss_values, fp, protocol=3)

    with open(experiment_directory + "/test_loss.pickle", "wb") as fp:
        pickle.dump(test_loss_values, fp, protocol=3)
    
    with open(experiment_directory + "/train_bce.pickle", "wb") as fp:
        pickle.dump(train_bce, fp, protocol=3)

    with open(experiment_directory + "/train_kld.pickle", "wb") as fp:
        pickle.dump(train_kld, fp, protocol=3)

    with open(experiment_directory + "/test_bce.pickle", "wb") as fp:
        pickle.dump(test_bce, fp, protocol=3)

    with open(experiment_directory + "/test_kld.pickle", "wb") as fp:
        pickle.dump(test_kld, fp, protocol=3)

    # Save latent spaces in each epochs 
    targets_test = de_encoding(targets_test, dict_encoding)
    df_test = pd.DataFrame({'label':targets_test, 'z1':mu_test[:,0], 'z2':mu_test[:,1]})
    df_test.to_csv(experiment_directory + "/latent_epoch"+str(epoch)+".csv")
    print(f"--> Epoch {epoch}: Saved latent values")

    # Save the model if in epochs_save
    if epoch in epochs_save:
        torch.save(model, experiment_directory + "/model_" + str(epoch) + ".pt")

# Close Tensorboard writer
writer.flush()
writer.close()


# Pickle dump TRAIN metrics per EPOCH
with open(experiment_directory + "/train_loss_epoch.pickle", "wb") as fp:
    pickle.dump(train_loss_epoch_list, fp, protocol=3)

with open(experiment_directory + "/train_bce_epoch.pickle", "wb") as fp:
    pickle.dump(train_bce_epoch_list, fp, protocol=3)

with open(experiment_directory + "/train_kld_epoch.pickle", "wb") as fp:
    pickle.dump(train_kld_epoch_list, fp, protocol=3)

# Pickle dump TEST metrics per EPOCH
with open(experiment_directory + "/test_loss_epoch.pickle", "wb") as fp: 
    pickle.dump(test_loss_epoch_list, fp, protocol=3)

with open(experiment_directory + "/test_bce_epoch.pickle", "wb") as fp:
    pickle.dump(test_bce_epoch_list, fp, protocol=3)

with open(experiment_directory + "/test_kld_epoch.pickle", "wb") as fp:
    pickle.dump(test_kld_epoch_list, fp, protocol=3)
