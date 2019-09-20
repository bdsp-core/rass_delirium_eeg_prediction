import numpy as np
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as FF
from ordistic_regression import OrdisticRegression


class ResBlock2d(nn.Module):
    def __init__(self, Lin, Lout, filter_len, dropout, subsampling, momentum, act='relu', maxpool_padding=0, deconv_padding=None, bn_affine=True, transpose=False, residual=True):
        assert filter_len%2==1
        super(ResBlock2d, self).__init__()
        self.Lin = Lin
        self.Lout = Lout
        self.filter_len = filter_len
        self.dropout = dropout
        self.subsampling = subsampling
        self.momentum = momentum
        self.act = act
        self.maxpool_padding = maxpool_padding
        self.bn_affine = bn_affine
        self.transpose = transpose
        self.residual = residual
        if deconv_padding is None:
            self.deconv_padding = self.subsampling-1
        else:
            self.deconv_padding = deconv_padding

        self.bn1 = nn.BatchNorm2d(self.Lin, momentum=self.momentum, affine=self.bn_affine)
        if self.act=='leaky':
            self.relu1 = nn.LeakyReLU()
        else:
            self.relu1 = nn.ReLU()
        if self.dropout is not None:
            self.dropout1 = nn.Dropout(self.dropout)
        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(self.Lin, self.Lin, (self.filter_len,1), stride=(self.subsampling,1), padding=(self.filter_len//2,0), output_padding=(self.deconv_padding,0), bias=False)
        else:
            self.conv1 = nn.Conv2d(self.Lin, self.Lin, (self.filter_len,1), stride=(self.subsampling,1), padding=(self.filter_len//2,0), bias=False)
        self.bn2 = nn.BatchNorm2d(self.Lin, momentum=self.momentum, affine=self.bn_affine)
        if self.act=='leaky':
            self.relu2 = nn.LeakyReLU()
        else:
            self.relu2 = nn.ReLU()
        if self.dropout is not None:
            self.dropout2 = nn.Dropout(self.dropout)
        if self.transpose:
            self.conv2 = nn.ConvTranspose2d(self.Lin, self.Lout, (self.filter_len,1), stride=1, padding=(self.filter_len//2,0), bias=False)
        else:
            self.conv2 = nn.Conv2d(self.Lin, self.Lout, (self.filter_len,1), stride=1, padding=(self.filter_len//2,0), bias=False)
        #self.bn3 = nn.BatchNorm2d(self.Lout, momentum=self.momentum, affine=self.bn_affine)
        if self.Lin==self.Lout and self.subsampling>1:
            if self.transpose:
                self.pooling = nn.Upsample(scale_factor=(self.subsampling,1))#, size=)
            else:
                self.pooling = nn.MaxPool2d((self.subsampling,1), padding=(self.maxpool_padding,0))

    def forward(self, x):
        if self.Lin==self.Lout:
            res = x
        x = self.bn1(x)
        x = self.relu1(x)
        if self.dropout is not None:
            x = self.dropout1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout is not None:
            x = self.dropout2(x)
        x = self.conv2(x)
        if self.residual and self.Lin==self.Lout:
            if self.subsampling>1:
                x = x+self.pooling(res)
            else:
                x = x+res
        #x = self.bn3(x)
        return x
        
        
class EEGNet_CNN(nn.Module):
    """
    """
    def __init__(self, output_type, K, prior=None):
        super(EEGNet_CNN, self).__init__()
        self.output_type = output_type
        self.n_channel = 2
        self.cnn_filter_num = 32
        self.cnn_filter_len = 9#17
        self.padding = self.cnn_filter_len//2
        self.dropout = 0.1
        self.momentum = 0.1
        self.subsampling = 4
        self.K = K
        self.prior = prior
        self.drug_num = 0#drug_num
        self.bn_affine = False
        self.Hdim = 3*self.cnn_filter_num*2#self.cnn_bottleneck+self.spec_bottleneck
        #self.Hdim_trainable = self.Hdim
        
        #(nch, 250)
        #self.conv1_time = nn.Conv3d(1, self.cnn_filter_num, (self.cnn_filter_len, 1, 1), stride=1, padding=(self.padding,0,0), bias=True)
        #self.conv1_spat = nn.Conv3d(self.cnn_filter_num, self.cnn_filter_num, (1, self.n_channel, 1), padding=0, stride=1, bias=False)
        self.first_layer = nn.Conv2d(self.n_channel, self.cnn_filter_num, (self.cnn_filter_len,1), stride=1, padding=(self.padding,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.cnn_filter_num, momentum=self.momentum, affine=self.bn_affine)
        self.relu1 = nn.ReLU()
        
        #(F, 250)
        self.conv2_1 = nn.Conv2d(self.cnn_filter_num, self.cnn_filter_num, (self.cnn_filter_len,1), stride=(self.subsampling,1), padding=(self.padding,0), bias=False)
        self.bn2 = nn.BatchNorm2d(self.cnn_filter_num, momentum=self.momentum, affine=self.bn_affine)
        self.relu2 = nn.ReLU()
        if self.dropout is not None:
            self.dropout2 = nn.Dropout(self.dropout)
        self.conv2_2 = nn.Conv2d(self.cnn_filter_num, self.cnn_filter_num, (self.cnn_filter_len,1), stride=1, padding=(self.padding,0), bias=False)
        self.maxpool2 = nn.MaxPool2d((self.subsampling,1), padding=(1,0))
        #self.bn_input = nn.BatchNorm2d(self.cnn_filter_num, momentum=self.momentum, affine=self.bn_affine)
        #(F, 63)

        self.resblock1 = ResBlock2d(self.cnn_filter_num, self.cnn_filter_num, self.cnn_filter_len,
                self.dropout, 1, self.momentum, bn_affine=self.bn_affine)
        #(F, 63)
        self.resblock2 = ResBlock2d(self.cnn_filter_num, self.cnn_filter_num, self.cnn_filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2, bn_affine=self.bn_affine)
        #(F, 16)
        self.resblock3 = ResBlock2d(self.cnn_filter_num, self.cnn_filter_num*2, self.cnn_filter_len,
                self.dropout, 1, self.momentum, bn_affine=self.bn_affine)
        #(2*F, 16)
        self.resblock4 = ResBlock2d(self.cnn_filter_num*2, self.cnn_filter_num*2, self.cnn_filter_len,
                self.dropout, self.subsampling, self.momentum, bn_affine=self.bn_affine)
        #(2*F, 4)
        self.resblock5 = ResBlock2d(self.cnn_filter_num*2, self.cnn_filter_num*2, self.cnn_filter_len,
                self.dropout, 1, self.momentum, bn_affine=self.bn_affine)
        #(2*F, 4)
        self.resblock6 = ResBlock2d(self.cnn_filter_num*2, self.cnn_filter_num*2, self.cnn_filter_len,
                self.dropout, 2, self.momentum, bn_affine=self.bn_affine)
        #(2*F, 2)
        self.resblock7 = ResBlock2d(self.cnn_filter_num*2, self.cnn_filter_num*3, self.cnn_filter_len,
                self.dropout, 1, self.momentum, bn_affine=self.bn_affine)
        #(3*F, 2)
        self.bn_output = nn.BatchNorm2d(self.cnn_filter_num*3, momentum=self.momentum, affine=self.bn_affine)
        self.relu_output = nn.ReLU()
    
        if self.output_type=='mse':
            self.output_layer = nn.Linear(self.Hdim, 1)#4)
            self.output_act = None  # for training
            self.output_act2 = None # for testing
        elif self.output_type=='bin':
            self.output_layer = nn.Linear(self.Hdim, 1)#4)
            self.output_act = None
            self.output_act2 = nn.Sigmoid()
        elif self.output_type=='ce':
            self.output_layer = nn.Linear(self.Hdim, self.K)
            self.output_act = None
            self.output_act2 = nn.Softmax(dim=-1)
        elif self.output_type=='ordinal':
            self.output_layer = OrdisticRegression(self.Hdim, self.K, prior=self.prior)
            self.output_act = None
            self.output_act2 = th.exp
        self.init()
            
    @property
    def n_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_dropout(self, dropout):
        self.dropout = dropout
        self.resblock1.dropout = dropout
        self.resblock2.dropout = dropout
        self.resblock3.dropout = dropout
        self.resblock4.dropout = dropout
        self.resblock5.dropout = dropout
        self.resblock6.dropout = dropout
        self.resblock7.dropout = dropout

    def init(self, method='orth', random_state=2018):
        np.random.seed(random_state)
        th.manual_seed(random_state)
        if th.cuda.is_available():
            th.cuda.manual_seed(random_state)
            th.cuda.manual_seed_all(random_state)
            
        for wn, w in self.named_parameters():
            if not w.requires_grad:
                continue
            if 'weight' in wn.lower():
                if len(w.size())>=2:
                    if method=='orth':
                        nn.init.orthogonal(w)
                    else:
                        nn.init.xavier_normal(w)
                else:
                    nn.init.normal(w, std=0.01)
            elif 'bias' in wn.lower():
                nn.init.constant(w, 0.)
    
    def train(self, mode=True):
        for m in self.children():
            m.train(mode)
        self.training = mode
        return self
        
    def cnn_forward(self, x):
        #x = th.unsqueeze(x, -1)
        #x = x.permute(0,4,3,2,1)
        #x = self.conv1_time(x)
        #x = self.conv1_spat(x)
        #x = th.squeeze(x, -2)
        x = self.first_layer(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        res = x
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout is not None:
            x = self.dropout2(x)
        x = self.conv2_2(x)
        x = x+self.maxpool2(res)

        #x = self.bn_input(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        
        x = self.bn_output(x)
        x = self.relu_output(x)
        
        return x

    def forward(self, x, **kwargs):
        return_ordinal_z = kwargs.get('return_ordinal_z', False)
            
        N = x.shape[0]
        ff_input = len(x.size())==3
        
        if ff_input:
            # to be compatible when combined with RNN
            # the last dimention is time step in RNN, which is always 1 here
            x = th.unsqueeze(x, -1)
            Hcnn = self.cnn_forward(x)
            Hcnn = Hcnn.view(N,-1)
            
            H = Hcnn
                
            if return_ordinal_z and self.output_type=='ordinal':
                output = self.output_layer.forward1d(H)
            else:
                output = self.output_layer(H)
                
        else:
            x = x.permute(0,2,3,1)
            Hcnn = self.cnn_forward(x)
            Hcnn = Hcnn.permute(0,3,1,2).contiguous()
            Hcnn = Hcnn.view(N, Hcnn.shape[1], -1)
            
            H = Hcnn
                
            N = H.shape[0]
            H = H.view(-1, H.shape[-1])
            if return_ordinal_z and self.output_type=='ordinal':
                output = self.output_layer.forward1d(H)
            else:
                output = self.output_layer(H)
            output = output.view(N, -1, output.shape[-1])
            H = H.view(N, -1, H.shape[-1])
            
        if self.training:
            if output is not None and hasattr(self, 'output_act') and self.output_act is not None and not return_ordinal_z:
                output = self.output_act(output)
        else:
            if output is not None and hasattr(self, 'output_act2') and self.output_act2 is not None and not return_ordinal_z:
                output = self.output_act2(output)
        
        return output, H

