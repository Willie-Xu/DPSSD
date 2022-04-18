from dssd.modeling import registry
from .dpnnet import dpn92

__all__ = ['build_backbone', 'dpn92']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
