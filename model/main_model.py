import copy
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from model.Unet_model import UNET


class UnetModel:
    """
        Model trained for surface defects
        and their respective annotations
    """

    def __init__(self, opt):
        self.in_channels = opt.in_channels
        self.num_class = opt.num_class
        self.model = UNET(self.in_channels, self.num_class).cuda()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=1e-4, centered=True, momentum=0.8)
        self.opt = opt
        self.loss_graph = {'train': [], 'val': []}

    def __call__(self, data):
        """
            Return model output when called
        """
        return self.model(data.cuda())

    def train_net(self, dataloader):
        """
            Trains the network
        """
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')

        # - Print current training info
        print("U-Net model training ... ")

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 100.0

        for epoch in range(self.opt.num_epochs):
            print("*"*20)
            print("*   EPOCH : ", epoch)
            print("*"*20)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                epoch_loss = 0.0
                for data, label in dataloader[phase]:
                    data = data.cuda()
                    label = label.cuda()
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only training
                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(data)
                        loss = self.criterion(output, label)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * data.size(0)

                epoch_loss = running_loss / self.opt.dataset_size[phase]
                print("{0}\t EPOCH LOSS: {1:.5f}".format(
                    phase.upper(), epoch_loss))
                self.loss_graph[phase].append(epoch_loss)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print()
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), './weights/u_net_wts.pth')

    def save_result(self):
        """
            Saves Loss Graph
        """
        plt.plot(self.loss_graph['train'], label='train')
        plt.plot(self.loss_graph['val'], label='validation')
        plt.title('Loss Graph')
        plt.xlabel('No. of Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig('./results/loss_graph.png')
