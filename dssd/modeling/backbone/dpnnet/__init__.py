from ... import registry
from . import dpn

__all__ = ['dpn92']


@registry.BACKBONES.register('dpn92')
def dpn92(cfg, pretrained=True):
    return dpn.dpn92(pretrained=pretrained)
