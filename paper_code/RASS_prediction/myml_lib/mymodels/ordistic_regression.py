import numpy as np
from sklearn.utils.extmath import softmax
import torch as th
from torch import nn, optim
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class OrdisticRegression(nn.Module):
    """
    Rennie, J.D. and Srebro, N., 2005, July.
    Loss functions for preference levels: Regression with discrete ordered labels.
    In Proceedings of the IJCAI multidisciplinary workshop on advances in preference handling (pp. 180-186)
    """
    def __init__(self, D, K, prior=None):
        super(OrdisticRegression, self).__init__()
        self.D = D
        assert K>=3; self.K = K
        if prior is None:
            self.prior = None#self.register_buffer('prior', th.ones(self.K))#/self.K
            #self.register_buffer('log_prior', th.zeros(self.K))
        else:
            raise NotImplementedError('prior')
            self.register_buffer('prior', th.Tensor(prior))
            self.register_buffer('log_prior', th.log(self.prior))
        
        self.linear = nn.Linear(self.D, 1, bias=True) # 1 because one output, although multiple thresholds
        
        # mu1 = -1, mu2 = tanh(itanh_mu2), mu3 = tanh(itanh_mu2+exp(log_diff_itanh_mus3)), ..., muK = 1
        self.register_buffer('mu1', -th.ones(1))
        self.register_buffer('muK', th.ones(1))
        
        init_mus = np.linspace(-1,1,self.K)
        self.itanh_mu2 = Parameter(th.Tensor([np.arctanh(init_mus[1])])) # itanh: inverse tanh
        if self.K>=4:
            self.log_diff_itanh_mus = Parameter(th.Tensor(np.log(np.diff(np.arctanh(init_mus[1:-1])))))
        
    def init(self):
        self.linear.reset_parameters()
    
    def forward(self, x):
        mus = self.get_mus(dtype='torch').view(1,-1)
        z = self.linear(x)        
        if self.prior is None:
            pi = F.log_softmax(-th.abs(mus-z), dim=1)
        else:
            pi = F.log_softmax(Variable(self.log_prior, requires_grad=False)-th.abs(mus-z), dim=1)
        return pi
    
    def forward1d(self, x):
        y = self.linear(x)
        return y
    
    def get_mus(self, dtype='numpy'):
        mus = [Variable(self.mu1, requires_grad=False), th.tanh(self.itanh_mu2)]
        if self.K>=4:
            mus.append(th.tanh(self.itanh_mu2 + th.cumsum(th.exp(self.log_diff_itanh_mus),0)))
        mus.append(Variable(self.muK, requires_grad=False))
        mus = th.cat(mus)
        
        if dtype=='numpy':
            mus = mus.data.cpu().numpy()
            
        return mus
    
    def get_proba(self, z):
        mus = self.get_mus()
        z = np.array(z).reshape(-1,1)
        if self.prior is None:
            log_prior = 0
        else:
            log_prior = self.log_prior
        return softmax(log_prior-np.abs(mus-z))

    def get_weight(self):
        for wn, w in self.named_parameters():
            if 'weight' in wn:
                yield w
                
    def get_bias(self):
        for wn, w in self.named_parameters():
            if 'bias' in wn:
                yield w
        
        
class MyMultiClassLoss(_Loss):
    """
    yp: N x K
    y: N
    returns [yp[y[0]], ..., yp[y[N-1]]]
    """
    def __init__(self, K, max_batch_size=128, size_average=True, reduce=True):
        super(MyMultiClassLoss, self).__init__(size_average)
        self.reduce = reduce
        self.K = K
        self.max_batch_size = max_batch_size
        self.register_buffer('increment', (th.arange(max_batch_size)*self.K).long())# when object.cuda(), also send to cuda
    
    def forward(self, input, target, weight=None):
        N = input.size(0)
        assert not target.requires_grad
        assert N==target.size(0) and 0<N<=self.max_batch_size
        if weight is not None:
            assert not weight.requires_grad
            assert N==weight.size(0)
        res = -th.take(input, target+Variable(self.increment[:N], requires_grad=False))
        
        if self.reduce:
            if weight is None:
                res = th.sum(res)
            else:
                res = th.sum(res*weight)
            if self.size_average:
                res = res/N
        return res
        
        
if __name__ == '__main__':
    import pdb
    from torch.utils.data import Dataset, DataLoader
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    
    np.random.seed(10)
    K = 3
    D = 1
    N = 1000
    epsilon = 1e-2
    X = np.random.randn(N,D)
    w = np.random.randn(D)
    b1 = np.random.randn()
    bs = np.r_[-np.inf, b1, b1+np.cumsum(np.exp(np.random.randn(K-2))), np.inf]
    yc = np.dot(X, w) + np.random.randn(N)*epsilon
    y1d = np.zeros(N)
    for i in range(K):
        ids = np.where((yc > bs[i]) & (yc <= bs[i+1]))[0]
        y1d[ids] = i
    
    lr = 0.001
    epochs = 100
    batch_size = 16
    n_jobs = 1
    use_gpu = True
    
    model = OrdisticRegression(D,K)
    loss_function = MyMultiClassLoss(K)
    #optimizer = optim.RMSprop([{'params': model.weight, 'weight_decay': 0.01},
    #                           {'params': model.bias1},
    #                           {'params':model.log_diff_bias}], lr=lr)
    optimizer = optim.RMSprop(filter(lambda p:p.requires_grad, model.parameters()), lr=lr)
    
    if use_gpu:
        model = model.cuda()
        loss_function = loss_function.cuda()
    
    class MyDataset(Dataset):
        def __init__(self, X, y):
            super(MyDataset, self).__init__()
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            X = self.X[idx]
            y = self.y[idx]
            return {'X': X.astype('float32'), 'y': y.astype(long)}
    
    Dtr = MyDataset(X[:800], y1d[:800])
    Dva = MyDataset(X[800:900], y1d[800:900])
    Dte = MyDataset(X[900:], y1d[900:])
    
    gen_tr = DataLoader(Dtr, batch_size=batch_size, shuffle=True,
                        num_workers=n_jobs, pin_memory=False)
    gen_va = DataLoader(Dva, batch_size=len(Dva), shuffle=False,
                        num_workers=n_jobs, pin_memory=False)
    gen_te = DataLoader(Dte, batch_size=len(Dte), shuffle=False,
                        num_workers=n_jobs, pin_memory=False)
    
    verbosity = 10
    model.train()
    for epoch in range(epochs):
        #running_loss = 0.
        for bi, batch in enumerate(gen_tr):
            X = Variable(batch['X'])
            y = Variable(batch['y'])
            if use_gpu:
                X = X.cuda()
                y = y.cuda()
            optimizer.zero_grad()

            yy = model(X)
            loss = loss_function(yy, y)
            loss.backward()
            
            optimizer.step()
            #running_loss += loss.data[0]
            #self.batch_loss_history.append(loss.data[0])
            #if bi % verbosity == verbosity-1:
            #    print('[%d, %d] loss: %g' % (epoch + 1, bi + 1, running_loss / verbosity))
            #    running_loss = 0.
                
        running_loss = 0.
        for bi, batch in enumerate(gen_va):
            X = Variable(batch['X'])
            y = Variable(batch['y'])
            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            yy = model(X)
            loss = loss_function(yy, y)
            running_loss += loss.data[0]
            print('[%d] %g'%(epoch+1,running_loss))
            
        running_loss = 0.
        for bi, batch in enumerate(gen_te):
            X = Variable(batch['X'])
            y = Variable(batch['y'])
            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            yy = model(X)
            loss = loss_function(yy, y)
            running_loss += loss.data[0]
            print('[%d] %g'%(epoch+1,running_loss))
    
    model.eval()
    pdb.set_trace()
    
    yptr = model(Dtr.X); ytr = Dtr.y
    ypva = model(Dva.X); yva = Dva.y
    ypte = model(Dte.X); yte = Dte.y
    print(spearmanr(ytr, yptr))
    print(spearmanr(yva, ypva))
    print(spearmanr(yte, ypte))
