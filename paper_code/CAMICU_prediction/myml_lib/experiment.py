import os
import os.path
import datetime
import timeit
import numpy as np
from sklearn.exceptions import NotFittedError
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from braindecode.torch_ext.util import np_to_var, var_to_np
from mymodels.ordistic_regression import MyMultiClassLoss


class MyStateFullDataLoader:
    def __init__(self, dataset, batch_size, random_state=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_state = random_state
        self.keys = self.dataset[0].keys()
        
        # create batch_ids so that the next batch is the continuation of the previous
        # [0,1,..,N] --> [3,1,...] (shuffled)
        N = dataset.len//dataset.shorten_amount
        shuffle_base_ids = np.arange(N)
        np.random.seed(self.random_state)
        np.random.shuffle(shuffle_base_ids)
        # [3,1,...] --> [3,1,.. | ... | ...] (split)
        self.batch_ids = np.split(shuffle_base_ids, np.arange(0,len(shuffle_base_ids),batch_size)[1:])
        if len(shuffle_base_ids)%batch_size==0:
            self.batch_ids[-1] = np.r_[self.batch_ids[-1], 0]
        # [3,1,.. | ... | ...] --> [[3,1,.. | ... | ...]
        #                           [3+N,1+N,.. | ... | ...]
        #                           [3+2N,1+2N,.. | ... | ...]] (tile)
        self.batch_ids = np.tile(self.batch_ids, (dataset.shorten_amount,1))+np.arange(dataset.shorten_amount).reshape(-1,1)*N
        if len(shuffle_base_ids)%batch_size==0:
            for xx in range(len(self.batch_ids)):
                self.batch_ids[xx][-1] = self.batch_ids[xx][-1][:-1]
        # [[3,1,.. | ... | ...]
        #  [3+N,1+N,.. | ... | ...]
        #  [3+2N,1+2N,.. | ... | ...]] --> [[3,3+N,3+2N...], [1,1+N,1+2N...], ...]
        self.batch_ids = self.batch_ids.T.flatten()
        
    def __iter__(self):
        for bids in self.batch_ids:
            batch = self.dataset[bids]
            yield {key:th.from_numpy(batch[key]) for key in self.keys}
            
            
class MyBalancedDataLoader:
    def __init__(self, dataset, batch_size, random_state=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_state = random_state
        self.keys = self.dataset[0].keys()
        
        #TODO assumes binary
        minor_class = 0 if np.sum(dataset.y[:,-1]==0)<np.sum(dataset.y[:,-1]==1) else 1
        major_class = 1-minor_class
        
        np.random.seed(self.random_state)
        minor_ids = np.random.choice(np.where(dataset.y[:,-1]==minor_class)[0], size=np.sum(dataset.y[:,-1]==major_class), replace=True)
        major_ids = np.where(dataset.y[:,-1]==major_class)[0]
        np.random.shuffle(major_ids)
        
        major_batch_ids = np.split(major_ids, np.arange(0,len(major_ids),batch_size//2)[1:])
        minor_batch_ids = np.split(minor_ids, np.arange(0,len(minor_ids),batch_size//2)[1:])
        self.batch_ids = [np.r_[major_batch_ids[ii], minor_batch_ids[ii]] for ii in range(len(major_batch_ids))]
        
    def __iter__(self):
        for bids in self.batch_ids:
            batch = self.dataset[bids]
            yield {key:th.from_numpy(batch[key]) for key in self.keys}
    

class Experiment(object):
    def __init__(self, model=None, batch_size=32, max_epoch=10, lr=0.001, label_smoothing_amount=None,
            optimizer=None, loss_function=None, scheduler=None, remember_best_metric='loss', clip_weight=None,
            regularizer=None, C=None, stateful=False, subsample=None,
            verbose=False, n_gpu=0, save_base_path='models', random_state=None):#, model_constraint=None, n_jobs=1
        self.model = model
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr = lr
        #self.n_jobs = n_jobs
        self.label_smoothing_amount = label_smoothing_amount
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.remember_best_metric = remember_best_metric
        self.clip_weight = clip_weight
        self.verbose = verbose
        self.n_gpu = n_gpu
        self.regularizer = regularizer
        self.C = C
        self.stateful = stateful
        self.subsample = subsample
        #self.model_constraint = model_constraint
        self.save_base_path = save_base_path
        self.random_state = random_state
        
    def get_weight_loss(self):
        loss = 0.
        if hasattr(self.model, 'cnn') and hasattr(self.model, 'rnn'):
            if type(self.C)!=dict:
                self.C = {'cnn':self.C,'rnn':self.C}
            if self.C.get('cnn',0)>0:
                for wn,w in self.model.cnn.named_parameters():
                    if w.requires_grad and 'weight' in wn.lower():
                        loss += th.sum(w**2)/th.numel(w)*self.C['cnn']
            if self.C.get('rnn',0)>0:
                for wn,w in self.model.rnn.named_parameters():
                    if w.requires_grad and 'weight' in wn.lower():
                        loss += th.sum(w**2)/th.numel(w)*self.C['rnn']
        else:
            if self.C is not None and self.C>0:
                for wn,w in self.model.named_parameters():
                    if w.requires_grad and 'weight' in wn.lower():
                        loss += th.sum(w**2)/th.numel(w)*self.C
        return loss
    
    def get_per_sample_loss(self, y, output):
        if type(self.loss_function)==MyMultiClassLoss:
            y = y[:,0]  # assumes y.shape = (x, 1)
            loss = self.loss_function(output, y)
            loss = loss.view(-1,1)
        elif self.loss_function=='mse':
            #loss = F.mse_loss(output, y, reduce=False)
            loss = (y - output)**2
            if len(loss.shape)>2:
                loss = loss.view(loss.shape[0],-1).mean(dim=1)
        elif self.loss_function=='ce':
            raise NotImplementedError(self.loss_function)
        elif self.loss_function=='bin':
            #loss = F.binary_cross_entropy_with_logits(yp, y, weight=Z, size_average=True)#reduce=False
            #if output.shape[1]>1:  # CAMICU multiple targets
            #    yp = output[:,:-1]
            #    max_val = (-yp).clamp(min=0)
            #    loss = yp - yp * y[:,:-1] + max_val + ((-max_val).exp() + (-yp - max_val).exp()).log()
            #else:
            max_val = (-output).clamp(min=0)
            loss = output - output * y + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()
            
        # TODO a separate function for regularization
        """
        if self.regularizer is not None:
            if 'activation_l1' in self.regularizer:
                hh = outputs[1]
                if hasattr(self.model, 'fixed_subnetwork') and self.model.fixed_subnetwork is not None:
                    hh = hh[:,:-self.model.fixed_subnetwork.Hdim]
                if self.C1 is not None:
                    penalty1 = th.abs(hh).mean(dim=1)
                    loss = loss + self.C1*penalty1
                if self.C2 is not None:
                    penalty2 = th.abs(hh[:,:self.model.cnn_bottleneck].mean(dim=1)-hh[:,self.model.cnn_bottleneck:].mean(dim=1))
                    loss = loss + self.C2*penalty2
            #elif self.regularizer=='activation_l2':
            #    penalty
            #    loss = loss + self.C*penalty
        """
        return loss
              
    def get_input_gradient(self, D):
        #if not self.fitted_:
        #    raise NotFittedError
        
        self.model.train()
        #TODO all bn/dropout.eval()
        
        # set requires_grad = False and record the old value
        #req_grad_old = {}
        #for name, param in self.model.named_parameters():
        #    req_grad_old[name] = param.requires_grad
        #    #param.requires_grad = False  # RuntimeError: 'inconsistent range for TensorList output'
            
        batch_size = 1
        gen = DataLoader(D, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
            
        gradients = []
        for bi, batch in enumerate(gen):
            X = Variable(batch['X'], requires_grad=True)
            
            if self.n_gpu>0:
                X = X.cuda()
            
            outputs = self.model(X, return_last=False, return_ordinal_z=True)
            outputs[0] = outputs[0][:,-1]
            if type(outputs) in [tuple,list]:
                output = outputs[0]
            else:
                output = outputs
            grad = th.autograd.grad(output, X)
            gradients.extend(grad[0].data.cpu().numpy())
        gradients = np.array(gradients)
       
        # restore to old value
        #for name, param in self.model.named_parameters():
        #    param.requires_grad = req_grad_old[name]
        self.model.eval()
        
        return gradients
              
    def run_one_epoch(self, epoch, dataset, train_or_eval, return_last=True, return_ordinal_z=False, use_gpu=False, evaluate_loss=True):
        if train_or_eval=='train':
            running_loss = 0.
            self.model.train()
            stateful = self.stateful
            subsample = self.subsample
            #samples_weight = th.from_numpy(dataset.Z[:,-1])
            #sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            if stateful:
                gen = MyStateFullDataLoader(dataset, batch_size=self.batch_size, random_state=epoch+2018)
            else:
                #gen = MyBalancedDataLoader(dataset, batch_size=self.batch_size, random_state=epoch+2018)
                gen = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)
            verbosity = 100
        else:
            total_loss = 0.
            total_outputs = []
            self.model.eval()
            stateful = False
            subsample = None
            #sampler = None
            gen = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
        N = 0.
        last_state = None
        mix_period = 20
                                
        for bi, batch in enumerate(gen):
            if subsample is not None and np.random.rand()>subsample:
                continue
            X = Variable(batch['X'])
            
            batch_size = len(X)
            N += batch_size
            
            y = Variable(batch['y'])
            if type(self.loss_function)==MyMultiClassLoss:
                y = y.long()
            if len(y.shape)==2:
                y = y[:,-1].contiguous()
            y = y.view(-1,1)
            
            Z = Variable(batch['Z'])
            if len(Z.shape)==2:
                Z = Z[:,-1].contiguous()
            Z = Z.view(-1,1)
            
            exist_L = 'L' in batch
            if exist_L:  # lengths
                #L = Variable(batch['L'])
                Lint = batch['L'].numpy()
                return_last2 = return_last
                return_last = False
            
            if use_gpu:
                X = X.cuda()
                y = y.cuda()
                Z = Z.cuda()
                #if exist_L:
                #    L = L.cuda()
            
            if train_or_eval=='train':
                self.optimizer.zero_grad()
            #if batch_size<4 and type(self.model)==nn.DataParallel: # this is a bug in nn.DataParallel if batch_size<n_gpu
            #    outputs = self.model(th.cat([X,X,X,X], dim=0), return_last=return_last, return_ordinal_z=return_ordinal_z)
            #    outputs = [outputs[iii][:batch_size] for iii in range(len(outputs))]
            #else:
            outputs = self.model(X, initial_state=last_state, return_last=return_last, return_ordinal_z=return_ordinal_z)
                        
            if exist_L:
                return_last = return_last2
            #if not return_last:
            #    mix_period = min(50, dataset.X.shape[1]//5)  # only count time steps after mix_period
            if stateful:
                if (bi+1)%dataset.shorten_amount==0:
                    last_state = None
                else:
                    if type(outputs[-1])==tuple:
                        last_state = tuple([xx.detach() for xx in outputs[-1]])
                    else:
                        last_state = outputs[-1].detach()
                    #if bi%dataset.shorten_amount!=0:
                    #    mix_period = 0
                
            # decide y, output, Z (weight) for computing loss
            if outputs[0] is not None:
                if exist_L:
                    nonzero_ids = np.where(Lint>0)[0]#th.nonzero(L)[:,0]
                    if len(nonzero_ids)==0:
                        continue
                    y_loss = y[nonzero_ids]#th.index_select(y, 0, nonzero_ids)
                    Z_loss = Z[nonzero_ids]
                    Lint = Lint[nonzero_ids].tolist()
                    output_loss = outputs[0][nonzero_ids]
                    ll = len(y_loss)
                    if return_last:
                        output_loss = th.cat([output_loss[iii,Lint[iii]-1].unsqueeze(0) for iii in range(ll)], dim=0)#.view(-1,1)
                    else:
                        #y_loss = th.cat([y_loss[iii].expand(Lint[iii]-mix_period) for iii in range(ll)], dim=0).view(-1,1)
                        output_loss = th.cat([output_loss[iii,mix_period:Lint[iii]].mean(0).unsqueeze(0) for iii in range(ll)], dim=0)
                        #Z_loss = th.cat([Z_loss[iii].expand(Lint[iii]-mix_period) for iii in range(ll)], dim=0).view(-1,1)
                else:
                    if return_last:
                        y_loss = y
                        output_loss = outputs[0]
                        Z_loss = Z
                    else:
                        y_loss = y#.expand(batch_size,outputs[0].shape[1]-mix_period).contiguous().view(-1,1)
                        output_loss = outputs[0][:,mix_period:].mean(dim=1)#.contiguous(); output_loss=output_loss.view(-1,output_loss.shape[-1])
                        Z_loss = Z#.expand(batch_size,outputs[0].shape[1]-mix_period).contiguous().view(-1,1)
                # this is for autoencoder
                if self.loss_function=='mse' and len(output_loss.shape)>2:
                    y_loss = X
                    Z_loss = 1.
                
            if train_or_eval=='train':
                loss = self.get_per_sample_loss(y_loss, output_loss)
                #if len(loss.shape)>1 and loss.shape[1]>1:  # multiple labels
                #    loss = loss*Z[:,:-1]
                #    loss = loss.sum(dim=1)
                #else:
                loss = loss*Z_loss
                loss = th.mean(loss)+self.get_weight_loss()
                running_loss += float(loss.data.cpu().numpy())
                
                loss.backward()
                if self.clip_weight>0:
                    nn.utils.clip_grad_norm(self.model.parameters(), self.clip_weight)
                self.optimizer.step()

                if bi % verbosity == verbosity-1:
                    #print('\n'.join(['%s\t%f'%(wn,w.grad.data.max()-w.grad.data.min()) for wn,w in self.model.named_parameters() if w.requires_grad and w.grad is not None]))
                    #print('\n'.join(['%s\t%f'%(wn,th.mean(th.abs(w.grad.data))) for wn,w in self.model.named_parameters() if w.requires_grad and w.grad is not None]))
                    print('[%d, %d %s] loss: %g' % (epoch + 1, bi + 1, datetime.datetime.now(), running_loss / verbosity))
                    running_loss = 0.
            else:
                if evaluate_loss:
                    loss = self.get_per_sample_loss(y_loss, th.log(output_loss))#, matching_features=F)
                    #if len(loss.shape)>1 and loss.shape[1]>1:
                    #    loss = loss*Z[:,:-1]  # multiple labels
                    #else:
                    loss = loss*Z_loss
                    if not return_last:
                        loss = loss*batch_size/len(y_loss)
                    loss = th.sum(loss)+self.get_weight_loss()
                    total_loss += float(loss.data.cpu().numpy())
                    
                outputs2 = []
                for ii in range(len(outputs)):
                    if outputs[ii] is None or type(outputs[ii])==tuple:
                        outputs2.append([])
                    else:
                        outputs2.append(var_to_np(outputs[ii]))
                total_outputs.append(outputs2)
                
            del outputs
            
        if train_or_eval!='train':
            if N==0:
                N=1
            return total_loss/N, total_outputs
    
    def fit(self, Dtr, Dva=None, Dte=None, Dtrva=None, return_last=True, init=False, suffix=''):
        self.fitted_ = False
        if self.n_gpu>0:
            self.model = self.model.cuda()
            if self.loss_function=='ordinal':
                self.loss_function = MyMultiClassLoss(Dtr.K, reduce=False)
            if isinstance(self.loss_function, nn.Module):
                self.loss_function = self.loss_function.cuda()
                
        if init:
            self.model.init(self.random_state)
            
        np.random.seed(self.random_state)
        th.manual_seed(self.random_state)
        if th.cuda.is_available() and self.n_gpu>0:
            th.cuda.manual_seed(self.random_state)
            th.cuda.manual_seed_all(self.random_state)
        
        if self.n_gpu>1:
            #raise NotImplementedError
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))
            
        self.optimizer = optim.RMSprop(filter(lambda p:p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = None#ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, verbose=True, patience=2)
        
        self.train_history = {'loss':[]}
        if Dva is not None:
            self.train_history['valid_loss'] = []
        if Dte is not None:
            self.train_history['test_loss'] = []
        self.best_perf = np.inf
        self.best_epoch = self.max_epoch
        self.save_path = os.path.join(self.save_base_path,'current_best_model%s.pth'%suffix)
        self.initial_path = os.path.join(self.save_base_path,'initial_model%s.pth'%suffix)
        #if self.n_jobs==-1:
        #    self.n_jobs = multiprocessing.cpu_count()
            
        # save initial model before training
        if not os.path.exists(self.save_base_path):
            os.mkdir(self.save_base_path)
        self.save(self.initial_path)
            
        st = timeit.default_timer()
        for epoch in range(self.max_epoch):
            if self.scheduler is not None and type(self.scheduler)!=ReduceLROnPlateau:
                self.scheduler.step()
            self.run_one_epoch(epoch, Dtr, 'train', use_gpu=self.n_gpu>0, return_last=return_last)

            if Dva is not None:
                val_loss = self.evaluate(Dva, use_gpu=self.n_gpu>0, return_last=return_last)#,'auc'
                current_perf = val_loss[self.remember_best_metric]
                self.train_history['valid_loss'].append(current_perf)
                print('[%d %s] val_loss: %g, current best: [epoch %d] %g' % (epoch + 1, datetime.datetime.now(), current_perf, np.argmin(self.train_history['valid_loss'])+1, np.min(self.train_history['valid_loss'])))
                
                if current_perf < self.best_perf:
                    self.best_epoch = epoch+1
                    self.best_perf = current_perf
                    if not os.path.exists(self.save_base_path):
                        os.mkdir(self.save_base_path)
                    self.save()
                    if Dte is not None:
                        te_loss = self.evaluate(Dte, use_gpu=self.n_gpu>0, return_last=return_last)#,'auc'
                        self.train_history['test_loss'].append(te_loss[self.remember_best_metric])
                        print('[%d %s] te_loss: %g' % (epoch + 1, datetime.datetime.now(), self.train_history['test_loss'][-1]))
                        #ypte = self.predict(Dte)
                        #ypte = np.nanmean(ypte[:,30:],axis=1).flatten()
                        #print(np.c_[Dte.y1d, ypte])
                if self.scheduler is not None and type(self.scheduler)==ReduceLROnPlateau:
                    self.scheduler.step(current_perf)

        et = timeit.default_timer()
        self.train_time = et-st

        if self.verbose:
            print('best epoch: %d, best val perf: %g'%(self.best_epoch, self.best_perf))
            print('training time: %gs'%self.train_time)
            
        self.load()
        self.fitted_ = True
        return self

    def save(self, save_path=None):
        #if not self.fitted_:
        #    raise NotFittedError
        if save_path is None:
            save_path = self.save_path
        if type(self.model)==nn.DataParallel:
            th.save(self.model.module.state_dict(), save_path)
        else:
            th.save(self.model.state_dict(), save_path)

    def load(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        self.model.load_state_dict(th.load(save_path))
        if self.n_gpu>0:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        #self.fitted_ = True
    
    def predict(self, D, output_id=0, return_last=True, return_ordinal_z=False, return_only_loss=False, use_gpu=None):#, concatenate=True
        #if not self.fitted_:
        #    raise NotFittedError

        if use_gpu is None:
            use_gpu = self.n_gpu>0
        if use_gpu:
            self.model = self.model.cuda()
            if self.loss_function=='ordinal':
                self.loss_function = MyMultiClassLoss(D.K, reduce=False)
            if isinstance(self.loss_function, nn.Module):
                self.loss_function = self.loss_function.cuda()
        
        loss, outputs = self.run_one_epoch(0, D, 'eval', use_gpu=use_gpu, return_last=return_last, return_ordinal_z=return_ordinal_z, evaluate_loss=return_only_loss)#, evaluate_loss=return_only_loss
            
        if return_only_loss:
            return loss
        else:
            yps = []
            if not hasattr(output_id, '__iter__'):
                output_id = [output_id]
            for oi in output_id:
                yp = [oo[oi] for oo in outputs]
                yp = np.concatenate(yp, axis=0)
                if yp.ndim==2 and yp.shape[1]==1:
                    yp = yp[:,0]
                yps.append(yp)
            if len(yps)==1:
                yps = yps[0]
            return yps
    
    def evaluate(self, D, use_gpu=None, return_last=True):
        #if not self.fitted_:
        #    raise NotFittedError
        perf = {}
        perf['loss'] = self.predict(D, return_only_loss=True, use_gpu=use_gpu, return_last=return_last)
        return perf
    
    def loglikelihood(self, D, use_gpu=None, reduce=True, return_last=True):
        yp = self.predict(D, use_gpu=use_gpu, return_last=return_last)
        if D.y.ndim==2:
            y = D.y[:,-1]
        else:
            y = D.y
            
        if self.model.output_type=='bin':
            ll = y*np.log(yp)+(1-y)*np.log(1-yp)
        else:
            raise NotImplementedError
            
        if reduce:
            return np.sum(ll)
        else:
            return ll

