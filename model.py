import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

##########################################################################
########################  Flatten Features  ##############################
##########################################################################
def flatten_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

# class NetRGB(nn.Module):
#     def __init__(self, size_data, n_classes=21, attention=False):
#         super(NetRGB, self).__init__()
#         self.attention = attention
#         # RGB
#         self.conv1_RGB = nn.Conv3d(3, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
#         self.pool1_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
#         size_data //= 2
#         if self.attention:
#             self.attention1_RGB = AttentionModule3D(30, 30, size_data, np.ceil(size_data/2), np.ceil(size_data/4))

#         self.conv2_RGB = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
#         self.pool2_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
#         size_data //= 2
#         if self.attention:
#             self.attention2_RGB = AttentionModule3D(60, 60, size_data, np.ceil(size_data/2), np.ceil(size_data/4))

#         self.conv3_RGB = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
#         self.pool3_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
#         size_data //= 2
#         if self.attention:
#             self.attention3_RGB = AttentionModule3D(80, 80, size_data, np.ceil(size_data/2), np.ceil(size_data/4))

#         self.linear_RGB = nn.Linear(80*size_data[0]*size_data[1]*size_data[2], 500) # 144000, 216000
#         self.relu_RGB = nn.ReLU()

#         # Fusion
#         self.linear = nn.Linear(500, n_classes)
#         self.final = nn.Softmax(1)

#     def forward(self, rgb, flow, save_outputs=False):
#         rgb = self.pool1_RGB(F.relu(self.conv1_RGB(rgb))) # rgb = self.pool1_RGB(F.relu(self.drop1_RGB(self.conv1_RGB(rgb))))
#         if self.attention:
#             rgb = self.attention1_RGB(rgb)

#         rgb = self.pool2_RGB(F.relu(self.conv2_RGB(rgb)))
#         if self.attention:
#             rgb = self.attention2_RGB(rgb)

#         rgb = self.pool3_RGB(F.relu(self.conv3_RGB(rgb)))
#         if self.attention:
#             rgb = self.attention3_RGB(rgb)

#         rgb = rgb.view(-1, flatten_features(rgb))
#         rgb = self.relu_RGB(self.linear_RGB(rgb))

#         rgb = self.linear(rgb)
#         label = self.final(rgb)

#         return label

###################################################################
####################### 3D Attention Model  #######################
###################################################################
class BlockConvReluPool3D(nn.Module):
    def __init__(self, in_dim, out_dim, conv_size=(3,3,3), conv_stride=(1,1,1), conv_padding=(1,1,1), pool_size=(2,2,2), pool_stride=(2,2,2), cuda=True):
        super(BlockConvReluPool3D, self).__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, conv_size, stride=conv_stride, padding=conv_padding)
        self.pool = nn.MaxPool3d(pool_size, stride=pool_stride)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, input):
        return self.pool(F.relu(self.conv(input)))

class ResidualBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, cuda=True):
        super(ResidualBlock3D, self).__init__()

        dim_conv = math.ceil(out_dim/4)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.bn1 = nn.BatchNorm3d(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_dim, dim_conv, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm3d(dim_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim_conv, dim_conv, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm3d(dim_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(dim_conv, out_dim, 1, 1, bias = False)
        self.conv4 = nn.Conv3d(in_dim, out_dim , 1, stride, bias = False)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, input):
        residual = input
        out = self.bn1(input)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.in_dim != self.out_dim) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out


class AttentionModule3D(nn.Module):
    def __init__(self, in_dim, out_dim, size1, size2, size3, cuda=True):
        super(AttentionModule3D, self).__init__()

        self.size1 = tuple(size1.astype(int))
        self.size2 = tuple(size2.astype(int))
        self.size3 = tuple(size3.astype(int))

        self.first_residual_blocks = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.trunk_branches = nn.Sequential(
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda),
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda)
        )

        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.skip1 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block2 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.skip2 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block3 = nn.Sequential(
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda),
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda)
        )

        self.block4 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.block5 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.block6 = nn.Sequential(
        	nn.BatchNorm3d(out_dim),
        	nn.ReLU(inplace=True),
        	nn.Conv3d(out_dim, out_dim , kernel_size = 1, stride = 1, bias = False),
        	nn.BatchNorm3d(out_dim),
        	nn.ReLU(inplace=True),
        	nn.Conv3d(out_dim, out_dim , kernel_size = 1, stride = 1, bias = False),
        	nn.Sigmoid()
        )

        self.final = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        ## Use GPU
        if cuda:
            self.cuda()


    def forward(self, input):
        input = self.first_residual_blocks(input)
        out_trunk = self.trunk_branches(input)

        # 1st level
        out_pool1 =  self.pool1(input)
        out_block1 = self.block1(out_pool1)
        out_skip1 = self.skip1(out_block1)

        #2sd level
        out_pool2 = self.pool2(out_block1)
        out_block2 = self.block2(out_pool2)
        out_skip2 = self.skip2(out_block2)

        # 3rd level
        out_pool3 = self.pool3(out_block2)
        out_block3 = self.block3(out_pool3)
        out_interp3 = F.interpolate(out_block3, size=self.size3, mode='trilinear', align_corners=True)
        out = out_interp3 + out_skip2

        #4th level
        out_softmax4 = self.block4(out)
        out_interp2 = F.interpolate(out_softmax4, size=self.size2, mode='trilinear', align_corners=True)
        out = out_interp2 + out_skip1

        #5th level
        out_block5 = self.block5(out)
        out_interp1 = F.interpolate(out_block5, size=self.size1, mode='trilinear', align_corners=True)

        #6th level
        out_block6 = self.block6(out_interp1)
        out = (1 + out_block6) * out_trunk

        # Final with Attention added
        out_last = self.final(out)

        return out_last

class CCNAttentionNet(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[8,16,32,64,128,256], cuda=True):
        super(CCNAttentionNet, self).__init__()

        self.convs = []
        self.attentions = []
        for idx, out_dim in enumerate(filters):
            if idx < 2:
                pool_size = [2,2,1]
                pool_stride = [2,2,1]
            else:
                pool_size = [2,2,2]
                pool_stride = [2,2,2]
            self.convs.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            size_data //= pool_stride
            in_dim = out_dim
            self.attentions.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))


        self.linear1 = nn.Linear(size_data[0]*size_data[1]*size_data[2]*in_dim, 5)
        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(5, n_classes)
        self.final = nn.Softmax(1)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features):
        for layer, attention in zip(self.convs, self.attentions):
            features = layer(features)
            features = attention(features)
        features = features.view(-1, flatten_features(features))
        features = self.activation(self.linear1(features))
        features = self.linear2(features)
        return self.final(features)