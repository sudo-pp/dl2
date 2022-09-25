import numpy as np
import torch
import torch.nn.functional as F
from domains import *
import sys
sys.path.append('../../')
import dl2lib as dl2


def kl(p, log_p, log_q):
    return torch.sum(-p * log_q + p * log_p, dim=1)

def transform_network_output(o, network_output):
    if network_output == 'logits':
        pass
    elif network_output == 'prob':
        o = [F.softmax(zo) for zo in o]
    elif inetwork_output == 'logprob':
        o = [F.log_sofmtax(zo) for zo in o]
    return o


class Constraint:

    def eval_z(self, z_batches):
        if self.use_cuda:
            z_inputs = [torch.cuda.FloatTensor(z_batch) for z_batch in z_batches]
        else:
            z_inputs = [torch.FloatTensor(z_batch) for z_batch in z_batches]

        for z_input in z_inputs:
            z_input.requires_grad_(True)
        z_outputs = [self.net(z_input) for z_input in z_inputs]
        for z_out in z_outputs:
            z_out.requires_grad_(True)
        return z_inputs, z_outputs

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        assert False

    def loss(self, x_batches, y_batches, z_batches, args):
        if z_batches is not None:
            z_inp, z_out = self.eval_z(z_batches)
        else:
            z_inp, z_out = None, None

        constr = self.get_condition(z_inp, z_out, x_batches, y_batches)
        
        neg_losses = dl2.Negate(constr).loss(args)
        pos_losses = constr.loss(args)
        sat = constr.satisfy(args)
            
        return neg_losses, pos_losses, sat, z_inp



I = {
  'plane': 0,
  'car': 1,
  'bird': 2,
  'cat': 3,
  'deer': 4,
  'dog': 5,
  'frog': 6,
  'horse': 7,
  'ship': 8,
  'truck': 9,
}


# class CifarDatasetConstraint(Constraint):

#     def __init__(self, net, margin, use_cuda=True, network_output='logits'):
#         self.net = net
#         self.network_output = network_output
#         self.margin = margin
#         self.use_cuda = use_cuda
#         self.n_tvars = 1
#         self.n_gvars = 0
#         self.name = 'CSimilarityT'

#     def params(self):
#         return {'delta': self.margin, 'network_output': self.network_output}

#     def get_condition(self, z_inp, z_out, x_batches, y_batches):
#         x_out = self.net(x_batches[0])
#         x_out = transform_network_output([x_out], self.network_output)[0]
#         targets = y_batches[0]

#         rules = []
#         rules.append(dl2.Implication(dl2.BoolConst(targets == I['car']), dl2.GEQ(x_out[:, I['truck']], x_out[:, I['dog']] + self.margin)))
#         rules.append(dl2.Implication(dl2.BoolConst(targets == I['deer']), dl2.GEQ(x_out[:, I['horse']], x_out[:, I['ship']] + self.margin)))
#         rules.append(dl2.Implication(dl2.BoolConst(targets == I['plane']), dl2.GEQ(x_out[:, I['ship']], x_out[:, I['frog']] + self.margin)))
#         rules.append(dl2.Implication(dl2.BoolConst(targets == I['dog']), dl2.GEQ(x_out[:, I['cat']], x_out[:, I['truck']] + self.margin)))
#         rules.append(dl2.Implication(dl2.BoolConst(targets == I['cat']), dl2.GEQ(x_out[:, I['dog']], x_out[:, I['car']] + self.margin)))
#         return dl2.And(rules)


class CifarConstraint(Constraint):

    def __init__(self, net, eps, margin, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.margin = margin
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'CSimilarityG'

    def params(self):
        return {'eps': self.eps, 'delta': self.margin, 'network_output': self.network_output}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]
    
    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        z_out = transform_network_output(z_out, self.network_output)[0]
        targets = y_batches[0]

        rules = []
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['car']), dl2.GEQ(z_out[:, I['truck']], z_out[:, I['dog']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['deer']), dl2.GEQ(z_out[:, I['horse']], z_out[:, I['ship']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['plane']), dl2.GEQ(z_out[:, I['ship']], z_out[:, I['frog']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['dog']), dl2.GEQ(z_out[:, I['cat']], z_out[:, I['truck']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['cat']), dl2.GEQ(z_out[:, I['dog']], z_out[:, I['car']] + self.margin)))
        return dl2.And(rules)
     

