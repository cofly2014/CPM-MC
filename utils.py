import torch
import torch.nn.functional as F
import os
import math
from enum import Enum
import sys
from torch.autograd import Variable
import numpy as np
class TestAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
#        self.current_best_accuracy_dict = {}
#        for dataset in self.datasets:
#            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

#    def is_better(self, accuracies_dict):
#        is_better = False
#        is_better_count = 0
#        for i, dataset in enumerate(self.datasets):
#            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
#                is_better_count += 1
#
#        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
#            is_better = True
#
#        return is_better

#    def replace(self, accuracies_dict):
#        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Test Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line

#    def get_current_best_accuracy_dict(self):
#        return self.current_best_accuracy_dict


def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    #elif test_mode:
    #    if not os.path.exists(checkpoint_dir):
    #        print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
    #        sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            #sys.exit()


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, resume, test_mode):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    #if not test_mode and not resume:
    # comment by guofei
    #if not resume:
        #os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')  # reduction='none'参数看是否聚合成标量
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    test_logits_sample = test_logits_sample.reshape(-1, test_logits_sample.shape[0], test_logits_sample.shape[1])
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())

def task_confusion(test_logits, test_labels, real_test_labels, batch_class_list):
    preds = torch.argmax(torch.logsumexp(test_logits, dim=0), dim=-1)
    real_preds = batch_class_list[preds]
    return real_preds

def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])

def timewise_cos(x,y):
    n,m,C,T=x.shape[:4]
    x=x.transpose(2,3)# C<->T
    y=y.transpose(2,3)# C<->T
    x=F.normalize(x.reshape(n,m,T,-1),dim=-1,p=2)  # x.reshape(n,m,T,-1)   [5，5，8，512]
    y=F.normalize(y.reshape(n,m,T,-1),dim=-1,p=2)
    dist=(1-(x*y).sum(-1)).sum(-1)
    return dist
def timewise_cos1(x,y):
    n, C, T = x.shape[:3]
    m = y.shape[0]
    x=x.transpose(1,2)# C<->T
    y=y.transpose(1,2)# C<->T
    all_distances_tensor = torch.zeros(m, n).cuda()
    for i in range(n):
        support = x[i].repeat(m,1,1,1,1)

        support=F.normalize(support.reshape(m,T,-1),dim=-1,p=2)  # x.reshape(n,m,T,-1)   [5，5，8，512]
        y=F.normalize(y.reshape(m,T,-1),dim=-1,p=2)
        dist=(1-(support*y).sum(-1)).sum(-1)
        all_distances_tensor[:,i] = dist
    return all_distances_tensor

def compute_optimal_transport_raw(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = torch.exp(- lam * M)
    # Avoiding poor math condition
    P = P / P.sum()
    u = torch.zeros(n).cuda()
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    while torch.max(torch.abs(u - P.sum(1))) > epsilon:
        # Shape (n, )
        u = P.sum(1)
        P = P * (r / u).reshape((-1, 1))
        P = P * (c / P.sum(0)).reshape((1, -1))
    return P, torch.sum(P * M)

def compute_optimal_transport(a,b, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    #M代价矩阵，pairwise为L2正则距离，结果为25※25
    M=F.pairwise_distance(a,b).cuda()
    n, m = M.shape
    r = Variable(torch.ones(n) / n).cuda()
    c = Variable(torch.ones(m) / m).cuda()
    P = Variable(torch.exp(- lam * M)).cuda()
    # Avoiding poor math condition
    P /= P.sum()
    u =torch.zeros(n).cuda()
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    while torch.max(torch.abs(u - P.sum(1))) >epsilon:
        # Shape (n, )
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    #P为OT最优方案，距离为P X M，此距离 加上L2距离，看情况比例
    return 0*torch.sum(P * M)+torch.sum(M,dim=1)


def mat_mat_l2_mult(mat,mat2):
    rows_norm = (torch.norm(mat, dim=1, p=2, keepdim=True)**2).repeat(1,mat2.shape[1])
    cols_norm = (torch.norm(mat2, dim=0, p=2, keepdim=True)**2).repeat(mat.shape[0], 1)
    rows_cols_dot_product = mat @ mat2
    ssd = rows_norm - 2*rows_cols_dot_product + cols_norm
    return ssd.sqrt()
'''
mat = torch.randn([20, 7])
mat2 = torch.randn([7,20])
re = mat_mat_l2_mult(mat, mat2)
print(re.shape)
print(mat_mat_l2_mult(mat, mat2))
'''
def EuclideanDistance(t1,t2):
    dim=len(t1.size())
    if dim==2:
        N,C=t1.size()
        M,_=t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(1, 0)) + torch.sum(t1 ** 2, -1).view(N, 1) + torch.sum(t2 ** 2, -1).view(1, M)
        #dist = dist + torch.sum(t1 ** 2, -1).view(N, 1)
        #dist = dist + torch.sum(t2 ** 2, -1).view(1, M)
        dist=torch.sqrt(dist)
        return Variable(dist)
    elif dim==3:
        B,N,_=t1.size()
        _,M,_=t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1)) + torch.sum(t1 ** 2, -1).view(B, N, 1) +  dist + torch.sum(t2 ** 2, -1).view(B, 1, M)
        #dist = dist + torch.sum(t1 ** 2, -1).view(B, N, 1)
        #dist = dist + torch.sum(t2 ** 2, -1).view(B, 1, M)
        dist=torch.sqrt(dist)
        return Variable(dist)
    else:
        print('error...')
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt