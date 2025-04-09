import numpy as np
import torch
from torch import nn

from nnunetv2.nets.UMambaEnc import UMambaEnc
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights

from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import AutoWeighted_DC_and_CE_and_Focal_loss
from nnunetv2.training.loss.sam import SAM
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform2

'''
{'input_channels': 4, 'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 320, 320], 'conv_op': <class 'torch.nn.modules.conv.Conv3d'>, 'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], 'num_classes': 3, 'deep_supervision': True, 'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'conv_bias': True, 'norm_op': <class 'torch.nn.modules.instancenorm.InstanceNorm3d'>, 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': <class 'torch.nn.modules.activation.LeakyReLU'>, 'nonlin_kwargs': {'inplace': True}}
{'input_channels': 4, 'n_stages': 6, 'features_per_stage': [32, 64, 128, 256, 320, 320], 'conv_op': <class 'torch.nn.modules.conv.Conv3d'>, 'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]], 'num_classes': 3, 'deep_supervision': True, 'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'conv_bias': True, 'norm_op': <class 'torch.nn.modules.instancenorm.InstanceNorm3d'>, 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': <class 'torch.nn.modules.activation.LeakyReLU'>, 'nonlin_kwargs': {'inplace': True}}
'''

MODEL_PARAMS = {
    'input_channels': 4,
    'n_stages': 6,
    'features_per_stage': [32, 64, 128, 256, 320, 320],
    'conv_op': nn.Conv3d,
    'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    'num_classes': 3,
    'deep_supervision': True,
    'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
    'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
    'conv_bias': True,
    'norm_op': nn.InstanceNorm3d,
    'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
    'dropout_op': None,
    'dropout_op_kwargs': None,
    'nonlin': nn.LeakyReLU,
    'nonlin_kwargs': {'inplace': True},
}
RUN_CONFIGURATION = {
    'deep_supervision_scales' : list(list(i) for i in 1 / np.cumprod(np.vstack(MODEL_PARAMS['strides']), axis=0))[:-1],
}

class UUMambaWrapper(nn.Module):
    # This class creates a new model using the UUMamba encoder model output and adds classification layers
    def __init__(self, uumambaModel, num_classes):  # removed inputShape parameter
        super(UUMambaWrapper, self).__init__()
        self.uumamba = uumambaModel
        # Classification head: global average pooling + two FC layers with ReLU in between
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        in_features = MODEL_PARAMS['features_per_stage'][-1]
        hidden_features = 128
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, num_classes)
    
    def forward(self, x):
        # Get encoder outputs; use only the last stage (bottleneck)
        features = self.uumamba(x)
        if isinstance(features, (list, tuple)):
            bottleneck = features[-1]
        else:
            bottleneck = features
        print('DEBUG: bottleneck shape', bottleneck.shape)
        pooled = self.global_pool(bottleneck)  # shape: (B, C, 1, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)
        print('DEBUG: flattened shape', flattened.shape)
        x = self.fc1(flattened)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_umamba_classification(initial_lr, weight_decay, num_epochs, device, num_classes=2, pretrained_path=None):
    full_model = UMambaEnc(**MODEL_PARAMS)
    full_model.apply(InitWeights_He(1e-2))
    full_model = full_model.to(device)
    if pretrained_path is not None:
        load_pretrained_weights(full_model, pretrained_path)
    encoder_model = full_model.encoder
    model = UUMambaWrapper(encoder_model, num_classes)
    model = model.to(device)
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, initial_lr, num_epochs)
    
    return model, loss, optimizer, lr_scheduler

def get_umamba(initial_lr, weight_decay, num_epochs, device, pretrained_path=None):
    model = UMambaEnc(**MODEL_PARAMS)

    model.apply(InitWeights_He(1e-2))
    model = model.to(device)
    # model = torch.compile(model)

    if pretrained_path is not None:
        load_pretrained_weights(model, pretrained_path)
        
    # Freeze model weights
    # for param in model.parameters():
    #     param.requires_grad = False

    # Training config

    def _build_loss():
        batch_dice_value = False
        
        if False:
            loss = DC_and_BCE_loss({},
                                    {'batch_dice': batch_dice_value, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
                                    use_ignore_label = None is not None, dice_class = MemoryEfficientSoftDiceLoss)
        else:
    #             loss = DC_and_CE_loss({'batch_dice': batch_dice_value,'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
    #                                   {},
    #                                   weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
            loss  = AutoWeighted_DC_and_CE_and_Focal_loss({'batch_dice': batch_dice_value, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
                                                            {},
                                                            {'alpha':0.5, 'gamma':2, 'smooth':1e-5},
                                                            ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if MODEL_PARAMS['deep_supervision']:
            weights = np.array([1 / (2**i) for i in range(len(RUN_CONFIGURATION['deep_supervision_scales']))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    loss = _build_loss()

    optimizer = SAM(
        params=[
            {'params': model.parameters(), 'weight_decay': weight_decay, 'lr': initial_lr},
            {'params': loss.loss.awl.parameters(), 'weight_decay': 0, 'lr': initial_lr}
        ],
        base_optimizer=torch.optim.SGD,
        momentum=0.99,
        nesterov=True,
        )
    lr_scheduler = PolyLRScheduler(optimizer.base_optimizer, initial_lr, num_epochs)

    return model, loss, optimizer, lr_scheduler

def transform_deep_supervision(batch):
    d = DownsampleSegForDSTransform2(RUN_CONFIGURATION['deep_supervision_scales'], 0, input_key='target', output_key='target')
    splits = d(target = batch.numpy())['target']
    return [torch.tensor(s) for s in splits]
