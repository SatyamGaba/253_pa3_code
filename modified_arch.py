import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pool = nn.MaxPool2d(2,stride = 2)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bnd1    = nn.BatchNorm2d(64)
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bnd2    = nn.BatchNorm2d(128)
        self.conv3   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation = 1)
        self.bnd3    = nn.BatchNorm2d(256)          
       
        self.relu    = nn.ReLU(inplace=True)
       
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation = 1,  output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation = 1 , output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,  dilation = 1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, 34, kernel_size=1)

    def forward(self, x):
        # Complete the forward function for the rest of the encoder
        x = self.bnd1(self.relu(self.conv1(x)))
        x = self.bnd1(self.pool(self.relu(self.conv1(x))))

        x = self.bnd2(self.relu(self.conv2(x)))
        x = self.bnd2(self.pool(self.relu(self.conv2(x))))

        x = self.bnd3(self.relu(self.conv3(x)))
        x = self.bnd3(self.relu(self.conv3(x)))                        
        out_encoder = self.bnd3(self.pool(self.relu(self.conv3(x))))

        x = self.bn1(self.relu(self.deconv1(out_encoder))) # ** = score in starter code

        # Complete the forward function for the rest of the decoder
        x = self.bn2(self.relu(self.deconv2(x)))
                     
        out_decoder = self.bn3(self.relu(self.deconv3(x)))
        score = self.classifier(out_decoder)                   
        
        # ***** might have to include softmax

        return score  # size=(N, n_class, x.H/1, x.W/1)x
