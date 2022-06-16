import torch
import cnine
from cnine import rtensor as _rtensor


class rtensor(torch.Tensor):

    @staticmethod
    def zeros(*args):
        return rtensor(torch.zeros(*args))
    
    @staticmethod
    def ones(*args):
        return rtensor(torch.ones(*args))
    
    @staticmethod
    def eye(*args):
        return rtensor(torch.eye(*args))
    
    @staticmethod
    def randn(*args):
        return rtensor(torch.randn(*args))

    def __str__(self):
        return _rtensor.view(self).__str__()

    def __repr__(self):
        return _rtensor.view(self).__repr__()



class ctensor(torch.Tensor):

    @staticmethod
    def zeros(*args):
        return ctensor(torch.zeros((2,)+args))
    
    @staticmethod
    def ones(*args):
        return ctensor(torch.ones((2,)+args))
    
    @staticmethod
    def eye(*args):
        return ctensor(torch.eye((2,)+args))
    
    @staticmethod
    def randn(*args):
        return ctensor(torch.randn((2,)+args))
    
    def __str__(self):
        return _ctensor.view(self).__str__()

    def __repr__(self):
        return _ctensor.view(self).__repr__()

