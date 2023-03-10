import os
import torch
import torch.nn as nn
from common_utils.bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
from skimage.morphology import label


class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        # torch.cuda.set_device(opt['gpu_ids'][0])
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_last_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        print(self.device)
        network = network.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt['find_unused_parameters']
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        else:
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.segNet)
        tidy_sequential(self.segNet)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.segNet)
        tidy_sequential(self.segNet)

    def remove_all_but_the_largest_connected_component(self, image: np.ndarray, for_which_classes: list,
                                                       minimum_valid_object_size: dict = None):
        """
        removes all but the largest connected component, individually for each class
        :param image:
        :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
        Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
        to use all foreground classes together)
        :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
        minimum_valid_object_size must match entries in for_which_classes
        :return:
        """
        if for_which_classes is None:
            for_which_classes = np.unique(image)
            for_which_classes = for_which_classes[for_which_classes > 0]

        assert 0 not in for_which_classes, "cannot remove background"
        largest_removed = {}
        kept_size = {}
        for c in for_which_classes:
            if isinstance(c, (list, tuple)):
                c = tuple(c)  # otherwise it cant be used as key in the dict
                mask = np.zeros_like(image, dtype=bool)
                for cl in c:
                    mask[image == cl] = True
            else:
                mask = image == c
            # get labelmap and number of objects
            lmap, num_objects = label(mask.astype(int), return_num=True)

            # collect object sizes
            object_sizes = {}
            for object_id in range(1, num_objects + 1):
                object_sizes[object_id] = (lmap == object_id).sum()

            largest_removed[c] = None
            kept_size[c] = None

            if num_objects > 0:
                # we always keep the largest object.
                maximum_size = max(object_sizes.values())
                kept_size[c] = maximum_size

                for object_id in range(1, num_objects + 1):
                    # we only remove objects that are relatively small
                    # if object_sizes[object_id] != maximum_size:
                    if object_sizes[object_id] < 0.5 * maximum_size:
                        # we only remove objects that are smaller than minimum_valid_object_size
                        remove = True
                        if minimum_valid_object_size is not None:
                            remove = object_sizes[object_id] < minimum_valid_object_size[c]
                        if remove:
                            image[(lmap == object_id) & mask] = 0
                            if largest_removed[c] is None:
                                largest_removed[c] = object_sizes[object_id]
                            else:
                                largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
        return image, largest_removed, kept_size
