"""Spectral projections derived offline from the training objects.

The competition's classes come in material pairs that shape cannot separate, and
the per-band Fisher separability of those pairs peaks on a *different* band for
each pair (banana on b7, orange on b10, car on b5), so no fixed choice of three
bands serves them all.

Rather than widen the model's input -- which costs the COCO-pretrained stem,
measured at -0.043 mAP in the first GPU round -- the 16 bands are projected down
to 3 by a discriminant learned from the annotated objects themselves. The stem
stays exactly as pretrained and still sees three channels; what changes is which
three. Mean material-pair separability of the channels handed to the stem:

    demo bands   [0, 1, 2]   1.227
    spread bands [0, 7, 15]  1.769
    best 3 bands [5, 6, 7]   2.180
    LDA 16 -> 3              3.786   (3.1x the demo)

This is the linear case of the 1x1 channel-adapter idea that recurs in the
multispectral-transfer literature, computed offline so it costs no GPU time and
adds no architectural risk.
"""

# Rows are output channels, columns are the 16 bands.
LDA_16_TO_3 = (
    (+0.820513, +0.843182, +0.198756, -0.250607, +0.011696, -0.502375, -0.492476, -1.000000, -0.569250, +0.706126, -0.250153, +0.043941, +0.099820, +0.401583, +0.002416, -0.063179),
    (+0.025261, +0.139606, -0.101773, -1.000000, +0.558753, -0.820136, +0.773573, -0.015368, -0.212180, +0.130058, +0.928383, +0.161274, -0.056674, -0.652883, -0.150852, +0.292906),
    (-0.077347, -0.545874, -0.709263, +0.626022, -0.203770, +0.192131, +0.933481, +0.007211, +0.069032, -1.000000, -0.356203, +0.652802, +0.445975, +0.166484, -0.157057, -0.043620),
)

# Highest mean per-pair Fisher separability among fixed triples.
BEST_BANDS = (5, 6, 7)
