import torch.nn as nn
import torchvision.models as models

class Resnet18(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad_ = False
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 34, kernel_size=1)

    def forward(self, x):
        # Complete the forward function for the rest of the encoder

        x = self.model.relu(self.model.bn1(self.model.conv1(x)))
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        out_encoder = self.model.layer4(x)

        x = self.bn1(self.relu(self.deconv1(out_encoder))) # ** = score in starter code

        # Complete the forward function for the rest of the decoder
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        out_decoder = self.bn5(self.relu(self.deconv5(x)))
        score = self.classifier(out_decoder)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
# rnet =Resnet18(3)
# print(rnet)