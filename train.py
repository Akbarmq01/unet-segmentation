import torch
from data.dataLoader import LoadData, Options
from model.main_model import UnetModel



# dataloader
Opt = Options()()
dataloaders = LoadData(Opt)()
# train
network = UnetModel(Opt)
network.train_net(dataloaders)