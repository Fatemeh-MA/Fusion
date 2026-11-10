from .baselines import CNNResidualSeg, DGCNNSeg, PointNetSeg
from .dualstream import DualStreamNetwork

MODELS = {
    "dualstream": DualStreamNetwork,    # FUSION, the network of the paper
    "pointnet": PointNetSeg,
    "dgcnn": DGCNNSeg,
    "cnn_residual": CNNResidualSeg,
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError("unknown model %r, choose from %s" % (name, sorted(MODELS)))
    return MODELS[name](**kwargs)


def output_layer(model):
    """The last layer, whose out_channels is the number of classes."""
    return getattr(model, model.LAST)
