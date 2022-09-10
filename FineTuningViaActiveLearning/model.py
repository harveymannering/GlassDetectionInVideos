import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


resnext_101_32x4d = nn.Sequential(  # Sequential,
    nn.Conv2d(4, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(256),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(512),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(1024),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(  # Sequential,
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                              nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                              nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                              nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                              nn.BatchNorm2d(1024),
                              nn.ReLU(),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                          nn.BatchNorm2d(2048),
                      ),
                      Lambda(lambda x: x),  # Identity,
                      ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.AvgPool2d((7, 7), (1, 1)),
    Lambda(lambda x: x.view(x.size(0), -1)),  # View,
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000)),  # Linear,
)

class ResNeXt101(nn.Module):
    def __init__(self, backbone_path):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d
        if backbone_path is not None:
            weights = torch.load(backbone_path)
            net.load_state_dict(weights, strict=True)
            print("Load ResNeXt Weights Succeed!")

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


class RefNet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefNet, self).__init__()
        # ---------------- Encoder ----------------------
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # ---------------- Decoder ----------------------
        self.deconv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv0 = nn.Conv2d(64, 4, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # ---------------- Encoder ----------------------
        hx = x
        hx = self.conv0(hx)

        hx1 = self.conv1(hx)
        hx = self.pool1(hx1)

        hx2 = self.conv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.conv5(hx)

        # ---------------- Decoder ----------------------
        hx = self.upsample(hx5)

        d4 = self.deconv4(torch.cat((hx, hx4), 1))
        hx = self.upsample(d4)

        d3 = self.deconv3(torch.cat((hx, hx3), 1))
        hx = self.upsample(d3)

        d2 = self.deconv2(torch.cat((hx, hx2), 1))
        hx = self.upsample(d2)

        d1 = self.deconv1(torch.cat((hx, hx1), 1))
        output = self.deconv0(d1)

        # output = self.fuse(torch.cat((ref, mask), 1))

        x0, ref = torch.split(output, [1, 3], 1)
        return x0, ref


# class ReflectionModule(nn.Module):
#     def __init__(self):
#         super(ReflectionModule, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, dilation=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=16, dilation=16),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=32, dilation=32),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, kernel_size=3, padding=64, dilation=64),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 3, kernel_size=1, padding=0, dilation=1),
#         )
#
#     def forward(self, x):
#         # ---------------- Encoder ----------------------
#         output = self.conv(x)
#         return output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, num_context=6):
        super(SELayer, self).__init__()
        self.channel = channel
        self.num_context = num_context
        self.context_channel = int(channel / num_context)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.context_attention = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, num_context, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, groups=num_context, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, groups=num_context, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        context_attention = self.context_attention(y)
        channel_attention = self.channel_attention(y)
        context_attention = context_attention.repeat(1, 1, self.context_channel, 1)
        context_attention = context_attention.view(-1, self.channel, 1, 1)
        attention = context_attention * channel_attention
        return x * attention.expand_as(x)

class DenseContrastModule(nn.Module):
    def __init__(self, planes):
        super(DenseContrastModule, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 8)

        self.local_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        self.SELayer = SELayer(int(self.inplanes / 8 * 6))

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_01 = local_1 - context_1

        context_2 = self.context_2(x)
        ccl_02 = local_1 - context_2

        context_3 = self.context_3(x)
        ccl_03 = local_1 - context_3

        ccl_12 = context_1 - context_2
        ccl_13 = context_1 - context_3
        ccl_23 = context_2 - context_3

        output = torch.cat((ccl_01, ccl_02, ccl_03, ccl_12, ccl_13, ccl_23), 1)
        output = self.SELayer(output)
        return output


class GlassNetMod(nn.Module):
    def __init__(self, backbone_path=None):
        super(GlassNetMod, self).__init__()

        resnext = ResNeXt101(backbone_path)
        self.prelayer = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.contrast_4 = DenseContrastModule(2048)
        self.contrast_3 = DenseContrastModule(1024)
        self.contrast_2 = DenseContrastModule(512)
        self.contrast_1 = DenseContrastModule(256)

        self.up_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1536, 768, 3, 1, 1),
            nn.BatchNorm2d(768),
            nn.ReLU())
        self.up_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(768, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU())
        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.up_0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4_predict = nn.Conv2d(768, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(384, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(192, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(96, 1, 3, 1, 1)

        self.refine = RefNet(69, 64)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        #prelayer= self.prelayer(x)
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        contrast_4 = self.contrast_4(layer4)
        up4 = self.up_4(contrast_4)
        layer4_predict = self.layer4_predict(up4)
        layer4_map = F.sigmoid(layer4_predict)

        contrast_3 = self.contrast_3(layer3 * layer4_map)
        up3 = self.up_3(contrast_3)
        layer3_predict = self.layer3_predict(up3)
        layer3_map = F.sigmoid(layer3_predict)

        contrast_2 = self.contrast_2(layer2 * layer3_map)
        up2 = self.up_2(contrast_2)
        layer2_predict = self.layer2_predict(up2)
        layer2_map = F.sigmoid(layer2_predict)

        contrast_1 = self.contrast_1(layer1 * layer2_map)
        up1 = self.up_1(contrast_1)
        layer1_predict = self.layer1_predict(up1)
        layer1_map = F.sigmoid(layer1_predict)
        layer1_map = F.upsample(layer1_map, size=x.size()[2:], mode='bilinear', align_corners=True)

        up0 = self.up_0(layer0)
        layer0_predict, ref = self.refine(torch.cat((x, up0, layer1_map), 1))

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        # if self.training:
        return layer0_predict, layer1_predict, layer2_predict, layer3_predict, layer4_predict, ref

        # return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
        #        F.sigmoid(layer1_predict)

    


    
