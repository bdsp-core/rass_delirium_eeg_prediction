import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as FF
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from ordistic_regression import OrdisticRegression

        
class EEGNet_RNN(nn.Module):
    def __init__(self, output_type, input_dim, K, model_type='lstm', rnn_hidden_num=64, rnn_dropout=0.5, rnn_layer_num=2, prior=None, fixed_subnetwork=None):#, max_batch_size=128
        super(EEGNet_RNN, self).__init__()
        self.output_type = output_type
        self.rnn_dropout = rnn_dropout
        self.rnn_hidden_num = rnn_hidden_num
        self.input_dim = input_dim
        self.K = K
        self.prior = prior
        self.fixed_subnetwork = fixed_subnetwork
        self.rnn_layer_num = rnn_layer_num
        self.bidirectional = False#bidirectional
        
        if type(fixed_subnetwork)==str:
            fixed_subnetwork = th.load(fixed_subnetwork)
        if type(fixed_subnetwork)==nn.DataParallel:
            fixed_subnetwork = fixed_subnetwork.module
        #if isinstance(fixed_subnetwork, EEGNet_RCNN):
        #    fixed_subnetwork = fixed_subnetwork.rnn
        self.fixed_subnetwork = fixed_subnetwork
        if self.fixed_subnetwork is not None:
            for p in self.fixed_subnetwork.parameters():
                p.requires_grad = False
            self.fixed_subnetwork.rnn.dropout = 0
        
        self.dropout_layer = nn.Dropout(rnn_dropout)
        if model_type.lower()=='lstm':
            RNN = nn.LSTM
        elif model_type.lower()=='gru':
            RNN = nn.GRU
        else:
            raise NotImplementedError(model_type)

        if type(self.rnn_hidden_num)==int:
            self.rnn1 = RNN(self.input_dim, self.rnn_hidden_num, self.rnn_layer_num,
                    bidirectional=self.bidirectional, dropout=self.rnn_dropout, batch_first=True)
        elif type(self.rnn_hidden_num) in [list, tuple]:
            assert len(self.rnn_hidden_num)==self.rnn_layer_num
            self.rnn1 = nn.ModuleList([
                        RNN(self.input_dim, self.rnn_hidden_num[i], 1, bidirectional=self.bidirectional,
                    dropout=self.rnn_dropout, batch_first=True) for i in range(self.rnn_layer_num)])
        
        if self.bidirectional:
            self.Hdim = self.rnn_hidden_num*2
        else:
            self.Hdim = self.rnn_hidden_num
            
        if self.fixed_subnetwork is not None:
            self.Hdim += self.fixed_subnetwork.rnn_hidden_num#TODO .Hdim
                    
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
            self.output_act2 = nn.Softmax(dim=1)
        elif self.output_type=='ordinal':
            self.output_layer = OrdisticRegression(self.Hdim, self.K, prior=self.prior)
            self.output_act = None
            self.output_act2 = th.exp
        self.init()
        
    @property
    def n_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def all_recurrent_weights(self):
        for wn, w in self.named_parameters():
            if 'rnn' in wn.lower() and 'weight' in wn.lower():
                yield w

    def init(self, method='orth', random_state=2018):
        np.random.seed(random_state)
        th.manual_seed(random_state)
        if th.cuda.is_available():
            th.cuda.manual_seed(random_state)
            th.cuda.manual_seed_all(random_state)
            
        for wn, w in self.named_parameters():
            if 'fixed_subnetwork' in wn.lower():
                continue
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
            #elif 'rnn' in wn.lower() and 'bias' in wn.lower():
            #    n = w.size(0)
            #    """LSTM
            #    w.data[:n//4].fill_(0.)
            #    w.data[n//4:n//2].fill_(1.)
            #    w.data[n//2:].fill_(0.)
            #    """
            #    #GRU
            #    w.data[:n//3].fill_(-1.)
            #    w.data[n//3:].fill_(0.)
            elif 'bias' in wn.lower():
                nn.init.constant(w, 0.)
        #n = self.rnn1.bias_ih_l0.size(0)
        #self.rnn1.bias_ih_l0.data[:n//3*2].fill_(1.)
        #self.rnn1.bias_ih_l0.data[n//3*2:].fill_(0.)
        #n = self.rnn1.bias_ih_l1.size(0)
        #self.rnn1.bias_ih_l1.data[:n//3*2].fill_(1.)
        #self.rnn1.bias_ih_l1.data[n//3*2:].fill_(0.)
                
    #def train(self, mode=True):
    #    for m in self.children():
    #        m.train(mode)
    #    if hasattr(self, 'fixed_subnetwork') and isinstance(self.fixed_subnetwork, nn.Module):
    #        for m in self.fixed_subnetwork.modules():
    #            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #                m.eval()
    #    return self

    def forward(self, x, initial_state=None, return_last=True, return_ordinal_z=False):
        N = x.size(0)
        if hasattr(self, 'dropout_layer'):
            x = self.dropout_layer(x)
        
        #if hasattr(self, 'fixed_subnetwork') and self.fixed_subnetwork is not None:

        if type(self.rnn_hidden_num) in [list, tuple]:
            if initial_state is None:
                for rnn in self.rnn1:
                    x, last_state = rnn(x)
                H = x
            else:
                raise NotImplementedError
                for i, rnn in enumerate(self.rnn1):
                    x, last_state = rnn(x, (initial_state[0][i], initial_state[1][i]))
                H = x
        else:
            if initial_state is None:
                H, last_state = self.rnn1(x)
            else:
                H, last_state = self.rnn1(x, initial_state)
        
        if return_last:
            H = H[:,-1]
            #H = H.mean(dim=1)
        else:
            H = H.contiguous().view(-1, H.size(2))
        
        if return_ordinal_z and self.output_type=='ordinal':
            output = self.output_layer.forward1d(H)
        else:
            output = self.output_layer(H)
            if self.training:
                if hasattr(self, 'output_act') and self.output_act is not None and not return_ordinal_z:
                    output = self.output_act(output)
            else:
                if hasattr(self, 'output_act2') and self.output_act2 is not None and not return_ordinal_z:
                    output = self.output_act2(output)
            
        if not return_last:
            output = output.view(N, -1, output.size(1))
            H = H.view(N, -1, H.size(1))

        return output, H, last_state
        
        
class ModuleSequence(nn.Module):
    def __init__(self, modules):
        super(ModuleSequence, self).__init__()
        self.modules = modules
        
    def cuda(self):
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].cuda()
        return self
        
    def forward(self, x, **kwargs):
        final_outputs = []
        for i in range(len(self.modules)):
            if i==0:
                xx = x
            else:
                xx = outputs[1]
            outputs = self.modules[i](xx, **kwargs)
            final_outputs = list(outputs)+final_outputs
        return final_outputs
        
    def train(self, mode=True):
        for i in range(len(self.modules)):
            self.modules[i].train(mode=mode)
    
