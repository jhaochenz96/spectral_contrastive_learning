from argparse import ArgumentError
import random
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.parallel
from torch.optim.lr_scheduler import LambdaLR
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", color=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.color = color
    
    def prCyan(self, skk): print("\033[96m {}\033[00m" .format(skk)) 
    def prPurple(self, skk): print("\033[95m {}\033[00m" .format(skk)) 

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.color == 'cyan':
            self.prCyan('\t'.join(entries))
        elif self.color == 'purple':
            self.prPurple('\t'.join(entries))
        else:
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #ipdb.set_trace()
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def spawn_processes(worker_fn, args, mpargs=None):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if mpargs is None:
        mpargs = (args, )
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(worker_fn, nprocs=ngpus_per_node, args=(ngpus_per_node, *mpargs))
    else:
        # Simply call main_worker function
        worker_fn(*(args.gpu, ngpus_per_node, *mpargs))

def init_proc_group(args, ngpus_per_node):
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + args.gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

def init_data_parallel(args, model, ngpus_per_node):
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model

def fix_dataparallel_keys(state_dict):
    # If a Dataparallel wrapped model was saved, remove the "module." prefix
    if all(key.startswith('module.') for key in state_dict):
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key[len('module.'):]] = val
        state_dict = new_state_dict
    return state_dict

def get_grad_norm(model, p=2):
    parameters = list(filter(lambda param: param.grad is not None, model.parameters()))
    return torch.norm(torch.stack([torch.norm(param.grad.detach(), p) for param in parameters]), p)

def grad_norm_for_loss(model, loss, grad_meter):
    model_grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        create_graph=False,
        only_inputs=True)
    grad_meter.update(torch.norm(torch.stack([torch.norm(m.detach()) for m in model_grads])))

def get_weight_norm(model, p=2):
    parameters = list(filter(lambda param: param.grad is not None, model.parameters()))
    return torch.norm(torch.stack([torch.norm(param.detach(), p) for param in parameters]), p)


def get_optimizer(optim_type, parameters, lr, wd, beta1=None, beta2=None, sgd_momentum=None):
    if optim_type == 'sgd':
        optim = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=sgd_momentum,
            weight_decay=wd)
    elif optim_type == 'adam':
        if wd != 0:
            print('should use adamw if wd > 0.')
        optim = torch.optim.Adam(
            parameters,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=wd)
    elif optim_type == 'adamw':
        optim = torch.optim.AdamW(
            parameters,
            lr = lr,
            betas=(beta1, beta2),
            weight_decay=wd)
    else:
        raise ArgumentError('invalid optimizer choice')

    return optim

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_lr_scheduler(lr_sched, optimizer, start_step, args):
    if lr_sched == 'fixed':
        lambda_fixed = lambda epoch: 1
        scheduler = LambdaLR(optimizer, lambda_fixed)
        return scheduler
    if lr_sched == 'cos':
        return get_cosine_schedule_with_warmup(
            optimizer, 0, args.max_iters, last_epoch=start_step, num_cycles=0.5)
