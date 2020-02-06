+import torch
import torch.nn as nn
import torch.nn.functional as F 


class Double_Conv(nn.Module):
	def __init__(self, in_channels,out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
						   nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
						   nn.BatchNorm2d(out_channels),
						   nn.ReLU(inplace=True),
						   nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1),
						   nn.BatchNorm2d(out_channels),
						   nn.ReLU(inplace=True)
						  )
	def forward(self, x):
		return self.double_conv(x)

class Maxpool_Double_Conv(nn.Module):
	def __init__(self, in_channels,out_channels):		
		super().__init__()
		self.maxpool_double_conv = nn.Sequential(
								   nn.MaxPool2d(kernel_size=2),
								   Double_Conv(in_channels,out_channels)
								  )
	def forward(self,x):
		return self.maxpool_double_conv(x)

class Upsample(nn.Module):
	def __init__(self, in_channels,out_channels):
		super().__init__()
		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.conv = Double_Conv(in_channels,out_channels)

	def forward(self,x1,x2):
		x1 = self.upsample(x1)
		# BCHW
		diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])
		diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
		x1 = F.pad(x1,[diff_x//2, diff_x - diff_x//2,diff_y//2, diff_y - diff_y//2])
		x = torch.cat([x1,x2],dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self,x):
		return self.conv(x)
