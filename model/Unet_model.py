from model_blocks import *
import torch.nn.functional as F 

class UNET(nn.Module):
	def __init__(self,input_channels,num_classes):
		super(UNET,self).__init__()
		self.input_channels = input_channels
		self.num_classes = num_classes

		self.double_conv1 = Double_Conv(self.input_channels, 64)
		self.down1 = Maxpool_Double_Conv(64, 128)
		self.down2 = Maxpool_Double_Conv(128, 256)
		self.down3 = Maxpool_Double_Conv(256, 512)
		self.down4 = Maxpool_Double_Conv(512,1024)
		self.up1 = Upsample(1024, 512)
		self.up2 = Upsample(512,256)
		self.up3 = Upsample(256,128)
		self.up4 = Upsample(128,64) 
		self.out_conv = OutConv(64, num_classes)

	def forward(self, x):
		c1 = self.double_conv1(x)
		c2 = self.down1(c1)
		c3 = self.down2(c2)
		c4 = self.down3(c3)
		c5 = self.down4(c4)
		# print("shape of c4 : ", c4.shape)
		# print("shape of c5 : ", c5.shape)
		x  = self.up1(c5,c4)
		x  = self.up2(x,c3)
		x =  self.up3(x, c2)
		x  = self.up4(x, c1)
		return self.out_conv(x)



# <----- test case for proper network output ----->

# input_image = torch.ones(1,3,512,512)
# net = UNET(3, 4)
# output = net(input_image)
# print(output.shape)


# input_image = torch.ones(1,3,500,600)
# net = UNET(3, 4)
# output = net(input_image)
# print(output.shape)

