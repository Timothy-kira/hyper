"""S3T: a spectral Transformer front end for HOD26, pretrained with MAE.

The spatial side of the detector is a COCO-pretrained RT-DETR; this package
holds the part that is ours -- band alignment, local-contrast priors, the
per-pixel spectral Transformer and its masked-autoencoder pretraining.
"""
