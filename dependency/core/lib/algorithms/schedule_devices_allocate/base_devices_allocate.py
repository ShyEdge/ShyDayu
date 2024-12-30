import abc

class BaseDevicesAllocate(metaclass=abc.ABCMeta):

    def __call__(self,scheduler):
        raise NotImplementedError
    