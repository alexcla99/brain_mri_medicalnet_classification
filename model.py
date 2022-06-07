from models import resnet

from collections import OrderedDict
from torch import nn
import torch

class _CustomDataParallel(nn.DataParallel):
    """Custom data parallel class."""

    # This class is used for multi-GPUs purpose only, it deals with following issues:
    # https://github.com/pytorch/pytorch/issues/43329
    # https://github.com/pytorch/pytorch/issues/16885

    def __init__(self, model, **kwargs):
        """Instanciate the custom data parallel class."""
        super(_CustomDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        """Rewrite the attribute link."""
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class MedicalNetClassifier(nn.Module):
    """Class for a modified ResNet with a classifier as the output."""

    def __init__(self, model_depth:int, batch_size:int, params:dict, checkpoint:dict, no_cuda:bool) -> None:
        """Instanciate a modified ResNet with specified parameters."""
        self.model_depth = model_depth
        self.batch_size = batch_size
        self.params = params
        self.checkpoint = checkpoint
        self.no_cuda = no_cuda
        super(MedicalNetClassifier, self).__init__()
        # Load the wanted ResNet model
        self.dense_factor = self.batch_size
        if self.model_depth == 10:
            self.dense_factor = 1
            self.model = resnet.resnet10(no_cuda=self.no_cuda)
        elif self.model_depth == 18:
            self.model = resnet.resnet18(no_cuda=self.no_cuda)
        elif self.model_depth == 34:
            self.model = resnet.resnet34(no_cuda=self.no_cuda)
        elif self.model_depth == 50:
            self.model = resnet.resnet50(no_cuda=self.no_cuda)
        elif self.model_depth == 101:
            self.model = resnet.resnet101(no_cuda=self.no_cuda)
        elif self.model_depth == 152:
            self.model = resnet.resnet152(no_cuda=self.no_cuda)
        elif self.model_depth == 200:
            self.model = resnet.resnet200(no_cuda=self.no_cuda)
        self.model.load_state_dict(self.checkpoint)
        # Freeze pre-trained model layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Add a classification output
        self.model.add_module("out_layer", self._output_block(self.params["dropout"]))
        if self.no_cuda == False:
            available_devices = torch.cuda.device_count()
            device_ids = [torch.cuda.device(i) for i in range(available_devices)]
            self.model = self.model.cuda()
            if available_devices == 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=device_ids
                )
            else:
                self.model = _CustomDataParallel(
                    self.model,
                    device_ids=device_ids
                )

    def _output_block(self, dropout:float) -> nn.Sequential:
        """Return an output block to be added at the end of a model."""
        output_block = nn.Sequential(OrderedDict([
            ("ad_avg_pool3d", nn.AdaptiveAvgPool3d(1)),
            ("flatten", nn.Flatten()),
            ("out_linear1", nn.Linear(self.dense_factor * 512, self.dense_factor * 256)),
            ("out_bn1", nn.BatchNorm1d(self.dense_factor * 256)),
            ("out_relu1", nn.ReLU(inplace=True)),
            ("out_linear2", nn.Linear(self.dense_factor * 256, 1)), # TODO output_size = num_classes
            ("out_bn2", nn.BatchNorm1d(1)),                         # TODO output_size = num_classes
            #("out_relu2", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(dropout)),
            ("classifier", nn.Softmax(dim=0))
        ]))
        return output_block

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward computations including the new block."""
        out = self.model.forward(x)
        out = self.model.out_layer(out)
        return out

def generate_model(model_depth:int, batch_size:int, params:dict, checkpoint:dict, no_cuda:bool) -> MedicalNetClassifier:
    """Build an dreturn the selected model."""
    model = MedicalNetClassifier(model_depth, batch_size, params, checkpoint, no_cuda)
    return model