import math
import torch
import torch.nn as nn
from collections import OrderedDict

# the YOLO network class
class YOLO(nn.Module):
    def __init__(self, num_boxes, num_classes, device='cpu'):
        super(YOLO, self).__init__()
        self.device = device
        # number of bounding boxes per cell (2 in our case)
        self.num_boxes = num_boxes
        # number of classes for detection (1 in our case: cracker box)
        self.num_classes = num_classes
        self.image_size = 448
        self.grid_size = 64
        # create the network
        self.network = self.create_modules()
        self.network.to(self.device)
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def create_modules(self):
        modules = nn.Sequential()

        # Convolutional Stack 1
        Conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        ReLU1 = nn.ReLU()
        MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_stack1 = nn.Sequential(OrderedDict([
            ('Conv1', Conv1),
            ('ReLU1', ReLU1),
            ('MaxPool1', MaxPool1)
        ]))

        # Convolutional Stack 2
        Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        ReLU2 = nn.ReLU()
        MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_stack2 = nn.Sequential(OrderedDict([
            ('Conv2', Conv2),
            ('ReLU2', ReLU2),
            ('MaxPool2', MaxPool2)
        ]))

        # Convolutional Stack 3
        Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        ReLU3 = nn.ReLU()
        MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_stack3 = nn.Sequential(OrderedDict([
            ('Conv3', Conv3),
            ('ReLU3', ReLU3),
            ('MaxPool3', MaxPool3)
        ]))

        # Convolutional Stack 4
        Conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        ReLU4 = nn.ReLU()
        MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_stack4 = nn.Sequential(OrderedDict([
            ('Conv4', Conv4),
            ('ReLU4', ReLU4),
            ('MaxPool4', MaxPool4)
        ]))

        # Convolutional Stack 5
        Conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        ReLU5 = nn.ReLU()
        MaxPool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_stack5 = nn.Sequential(OrderedDict([
            ('Conv5', Conv5),
            ('ReLU5', ReLU5),
            ('MaxPool5', MaxPool5)
        ]))

        # Convolutional Stack 6
        Conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        ReLU6 = nn.ReLU()
        MaxPool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_stack6 = nn.Sequential(OrderedDict([
            ('Conv6', Conv6),
            ('ReLU6', ReLU6),
            ('MaxPool6', MaxPool6)
        ]))

        # Convolutional Stack 7
        Conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        ReLU7 = nn.ReLU()
        conv_stack7 = nn.Sequential(OrderedDict([
            ('Conv7', Conv7),
            ('ReLU7', ReLU7),
        ]))

        # Convolutional Stack 8
        Conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        ReLU8 = nn.ReLU()
        conv_stack8 = nn.Sequential(OrderedDict([
            ('Conv8', Conv8),
            ('ReLU8', ReLU8),
        ]))

        # Convolutional Stack 9
        Conv9 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        ReLU9 = nn.ReLU()
        conv_stack9 = nn.Sequential(OrderedDict([
            ('Conv9', Conv9),
            ('ReLU9', ReLU9),
        ]))

        # Fully Connected Stack
        flatten = nn.Flatten()
        FC1 = nn.Linear(in_features=50176, out_features=256)
        FC2 = nn.Linear(in_features=256, out_features=256)
        output = nn.Linear(in_features=256, out_features=49*(5*self.num_boxes+self.num_classes))
        sigmoid = nn.Sigmoid()
        fc_stack = nn.Sequential(OrderedDict([
            ('Flatten', flatten),
            ('FC1', FC1),
            ('FC2', FC2),
            ('Output', output),
            ('Sigmoid', sigmoid)
        ]))

        modules.add_module('Conv Stack 1', conv_stack1)
        modules.add_module('Conv Stack 2', conv_stack2)
        modules.add_module('Conv Stack 3', conv_stack3)
        modules.add_module('Conv Stack 4', conv_stack4)
        modules.add_module('Conv Stack 5', conv_stack5)
        modules.add_module('Conv Stack 6', conv_stack6)
        modules.add_module('Conv Stack 7', conv_stack7)
        modules.add_module('Conv Stack 8', conv_stack8)
        modules.add_module('Conv Stack 9', conv_stack9)
        modules.add_module('FC Stack', fc_stack)
        return modules

    # output (batch_size, 5*B + C, 7, 7)
    # In the network output (cx, cy, w, h) are normalized to be [0, 1]
    # This function undo the noramlization to obtain the bounding boxes in the original image space
    def transform_predictions(self, output):
        batch_size = output.shape[0]
        x = torch.linspace(0, 384, steps=7)
        y = torch.linspace(0, 384, steps=7)
        corner_x, corner_y = torch.meshgrid(x, y, indexing='xy')
        corner_x = torch.unsqueeze(corner_x, dim=0)
        corner_y = torch.unsqueeze(corner_y, dim=0)
        corners = torch.cat((corner_x, corner_y), dim=0)
        # corners are top-left corners for each cell in the grid
        corners = corners.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
        pred_box = output.clone()

        # for each bounding box
        for i in range(self.num_boxes):
            # x and y
            pred_box[:, i*5, :, :] = corners[:, 0, :, :] + output[:, i*5, :, :] * self.grid_size
            pred_box[:, i*5+1, :, :] = corners[:, 1, :, :] + output[:, i*5+1, :, :] * self.grid_size
            # w and h
            pred_box[:, i*5+2, :, :] = output[:, i*5+2, :, :] * self.image_size
            pred_box[:, i*5+3, :, :] = output[:, i*5+3, :, :] * self.image_size

        return pred_box

    # forward pass of the YOLO network
    def forward(self, x):
        # raw output from the network
        output = self.network(x).reshape((-1, self.num_boxes * 5 + self.num_classes, 7, 7))
        # compute bounding boxes in the original image space
        pred_box = self.transform_predictions(output)
        return output, pred_box