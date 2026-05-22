
import torch
from torch import nn

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_units: An integer indicating number of input channels.
    output_units: An integer indicating number of output units.
    hidden_units: An integer indicating number of hidden units between layers.
  """
  def __init__(self,input_units:int,output_units:int,hidden_units:int) :
     super().__init__()

     self.block1 = nn.Sequential(
         nn.Conv2d(
             in_channels=input_units,
             out_channels=hidden_units,
             kernel_size=3,
             stride=1,
             padding=0
         ),
         nn.ReLU(),
         nn.Conv2d(
             in_channels=hidden_units,
             out_channels=hidden_units,
             kernel_size=3,
             stride=1,
             padding=0
         ),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2,stride=2)
     )

     self.block2 = nn.Sequential(
         nn.Conv2d(
             in_channels=hidden_units,
             out_channels=hidden_units,
             kernel_size=3,
             stride=1,
             padding=0
         ),
         nn.ReLU(),
         nn.Conv2d(
             in_channels=hidden_units,
             out_channels=hidden_units,
             kernel_size=3,
             stride=1,
             padding=0
         ),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2,stride=2)
     )

     self.classifier = nn.Sequential(
         nn.Flatten(),
         nn.Linear(
             in_features=hidden_units*13*13,
             out_features=output_units
         )
     )

  def forward(self,x:torch.Tensor):
    x = self.block1(x)
    x = self.block2(x)
    x = self.classifier(x)
    return x
