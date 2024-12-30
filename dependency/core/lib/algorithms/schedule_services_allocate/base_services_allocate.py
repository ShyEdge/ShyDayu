import abc

class BaseServicesAllocate(metaclass=abc.ABCMeta):

  def __call__(self,scheduler):
        raise NotImplementedError