import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image

def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                #full_path = os.path.join(path, file)
                filtered_files.append(file)

    return filtered_files



learning_map = {
  0 : 0,    # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

class_weights = { # as a ratio with the total number of points
  0: 0.018889854628292943,
  1: 0.0002937197336781505,
  10: 0.040818519255974316,
  11: 0.00016609538710764618,
  13: 2.7879693665067774e-05,
  15: 0.00039838616015114444,
  16: 0.0,
  18: 0.0020633612104619787,
  20: 0.0016218197275284021,
  30: 0.00017698551338515307,
  31: 1.1065903904919655e-08,
  32: 5.532951952459828e-09,
  40: 0.1987493871255525,
  44: 0.014717169549888214,
  48: 0.14392298360372,
  49: 0.0039048553037472045,
  50: 0.1326861944777486,
  51: 0.0723592229456223,
  52: 0.002395131480328884,
  60: 4.7084144280367186e-05,
  70: 0.26681502148037506,
  71: 0.006035012012626033,
  72: 0.07814222006271769,
  80: 0.002855498193863172,
  81: 0.0006155958086189918,
  99: 0.009923127583046915,
  252: 0.001789309418528068,
  253: 0.00012709999297008662,
  254: 0.00016059776092534436,
  255: 3.745553104802113e-05,
  256: 0.0,
  257: 0.00011351574470342043,
  258: 0.00010157861367183268,
  259: 4.3840131989471124e-05,
}

mapped_class_weights = []
for cls in range(20):
  weight = 0
  for key in learning_map:
    if learning_map[key]==cls:
      weight+=class_weights[key]
      total_weights+=class_weights[key]
  mapped_class_weights.append(1/weight)

weights = np.asarray(mapped_class_weights)
weights = torch.from_numpy(weights)


class_encoding={ 
  "unlabeled" : 0,
  "car" :1,
  "bicycle" : 2,
  "motorcycle" : 3,
  "truck" : 4,
  "other-vehicle":5,
  "person":6,
  "bicyclist":7,
  "motorcyclist":8,
  "road":9,
  "parking":10,
  "sidewalk":11,
  "other-ground":12,
  "building":13,
  "fence":14,
  "vegetation":15,
  "trunk":16,
  "terrain":17,
  "pole":18,
  "traffic-sign":19,
}

def map_labels(mask, lr_map = learning_map):
  new_mask=np.zeros(mask.shape)
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      new_mask[i,j]=lr_map[int(mask[i,j])]
  return new_mask

class RangeKitti(data.Dataset):
  
  def __init__(self, root_dir, mode='train'):

    train = ['00','01','02','03','04','05','06']
    test =  ['07','08','09','10']

    self.root_dir = root_dir

    if mode=='train':
      folders = train

    elif mode=='test':
      folders = test

    else :
      print("unkown mode")

    self.files = []
    for folder in folders:
      files = get_files(self.root_dir+'/'+folder+'/range')
      files = [self.root_dir+'/'+folder+'/range/'+file for file in files]
      self.files.extend(files)

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    path = self.files[index]
    proj = np.fromfile(path,dtype=np.int32).reshape(64,1024,6)
    proj = proj.astype(np.float32)/1000

    mask = map_labels(proj[:,:,5])

    proj = torch.from_numpy(proj[:,:,0:5])
    proj = torch.transpose(proj, 0,2)
    proj = torch.transpose(proj, 1,2)
    return proj, torch.from_numpy(mask).long()

"""# Metric"""

class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass

class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf



class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)

def train(model, data_loader, optim, criterion, metric, iteration_loss=False):
    
    model.train()
    epoch_loss = 0.0
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        # Get the inputs and labels
        inputs = batch_data[0].cuda()
        #print(batch_data[1].shape)
        labels = batch_data[1][:,::4,::4].cuda()
        # complete
        optim.zero_grad()
        # Forward propagation
        outputs = model(inputs).logits
        # ..........complete..........

        # Loss computation
        loss = criterion(outputs, labels)
        # ..........complete..........
      

        # Backpropagation
        loss.backward()
        optim.step()
        #optim.zero_grads()

        # ..........complete..........
 

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of the evaluation metric
        metric.add(outputs.detach(), labels.detach())

        if iteration_loss:
            print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

    return epoch_loss / len(data_loader), metric.value()

def test(model, data_loader, criterion, metric, iteration_loss=False):
    
    model.eval()
    epoch_loss = 0.0
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        # Get the inputs and labels
        inputs = batch_data[0].cuda()
        labels = batch_data[1][:,::4,::4].cuda()

        with torch.no_grad():
            # Forward propagation
            outputs = model(inputs).logits

            # Loss computation
            loss = criterion(outputs, labels)

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of evaluation the metric
        metric.add(outputs.detach(), labels.detach())

        if iteration_loss:
            print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
    for classe, res in zip(class_encoding, metric.value()[0]):
        print(f"[{classe}] : {res}")
    return epoch_loss / len(data_loader), metric.value()
    
    
def train_for_segformer(model, data_loader, optim, criterion, metric, iteration_loss=False):
    '''
    Training script: it allows the training of one epoch of the DNN.
    input:
      model: the model you want to train
      data_loader: the dataloader (the FIFO of data)
      optim: the optimizer you use
      criterion: the criterion you want to optimize
      metric: other criteria
      iteration_loss : boolean that allow you to print the loss
    output:
      epoch_loss: the loss of the full epoch
      metric.value(): the value of the other criteria
    '''  
    model.train()
    epoch_loss = 0.0
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        # Get the inputs and labels
        inputs = batch_data[0].cuda()
        #print(batch_data[1].shape)
        labels = batch_data[1][:,::4,::4].cuda()
        # complete
        optim.zero_grad()
        # Forward propagation
        outputs = model(inputs).logits
        # ..........complete..........

        # Loss computation
        loss = criterion(outputs, labels)
        # ..........complete..........
      

        # Backpropagation
        loss.backward()
        optim.step()
        #optim.zero_grads()

        # ..........complete..........
 

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of the evaluation metric
        metric.add(outputs.detach(), labels.detach())

        if iteration_loss:
            print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

    return epoch_loss / len(data_loader), metric.value()

def test_for_segformer(model, data_loader, criterion, metric, iteration_loss=False):
    '''
    Validation script: it allows the validationof the DNN.
    input:
      model: the DNN you want to train
      data_loader: the dataloader (the FIFO of data)
      criterion: the criterion you hav optimized
      metric: other criteria
      iteration_loss : boolean that allow you to print the loss
    output:
      epoch_loss: the loss of the full epoch
      metric.value(): the value of the other criteria
    '''  
    model.eval()
    epoch_loss = 0.0
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        # Get the inputs and labels
        inputs = batch_data[0].cuda()
        labels = batch_data[1][:,::4,::4].cuda()

        with torch.no_grad():
            # Forward propagation
            outputs = model(inputs).logits

            # Loss computation
            loss = criterion(outputs, labels)

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of evaluation the metric
        metric.add(outputs.detach(), labels.detach())

        if iteration_loss:
            print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
    for classe, res in zip(class_encoding, metric.value()[0]):
        print(f"[{classe}] : {res}")
    return epoch_loss / len(data_loader), metric.value()


