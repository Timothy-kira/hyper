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


# Penalised discriminant projections at three roughness levels, so the choice
# of starting point is measured rather than assumed. The penalty is on the
# second difference of the weights across bands, which is what makes a
# projection behave like a smooth sensor response curve instead of an
# arbitrary linear combination.
#
# Measured on 442 held-out objects: smoothing costs most of the separability it
# is meant to protect, and does not even buy noise robustness -- the projection
# already acts on an object's mean spectrum, so spatial averaging has removed
# the pixel noise a spectral filter would target.
#
#   penalty   sep(held-out)   roughness   sep @5% noise   sep @1-band shift
#   0.0             3.447        0.943           2.199               1.368
#   0.001           1.733        0.064           1.610               1.965
#   0.01            1.398        0.058           1.390               1.987
#
# Smoothing wins only against a whole-spectrum shift, and every frame here
# comes from one sensor, so band k is the same wavelength throughout. That is
# insurance against a risk this dataset does not carry -- hence 0.0 is the
# default, with the smoother starts left available to the search.
PDA_PROJECTIONS = {
    "0": (
        (
            0.8205134096177122,
            0.8431823645724326,
            0.1987562805008689,
            -0.2506067781294189,
            0.011695846962533863,
            -0.5023748530283455,
            -0.4924759139835989,
            -0.999999999997989,
            -0.5692504928269478,
            0.7061263434783711,
            -0.2501525174995029,
            0.04394141743727731,
            0.09981958355109528,
            0.401583349358285,
            0.002415995035606788,
            -0.06317936475657716
        ),
        (
            0.02526106412792499,
            0.13960609647729547,
            -0.10177290662552417,
            -0.9999999999979747,
            0.5587526868635889,
            -0.8201356770735837,
            0.7735730150208141,
            -0.015367630583825315,
            -0.21218045335570945,
            0.13005795679572113,
            0.928383453013779,
            0.16127351727953587,
            -0.056673583762502644,
            -0.6528830976886645,
            -0.15085180627470526,
            0.2929063614061286
        ),
        (
            -0.07734730537417482,
            -0.545874364053114,
            -0.7092627187654245,
            0.626022319643028,
            -0.20377009219446918,
            0.19213081728348888,
            0.9334811034728061,
            0.00721067552993082,
            0.06903174747778928,
            -0.9999999999980101,
            -0.35620250367268397,
            0.652802137457828,
            0.4459748884168419,
            0.16648370975979168,
            -0.1570573076848149,
            -0.04362016330674697
        )
    ),
    "0.001": (
        (
            0.9999999999982903,
            0.6604020282208848,
            0.33016839359106126,
            0.0266607061277293,
            -0.2302131095857345,
            -0.4217011649528806,
            -0.5306718263771192,
            -0.5485916235521019,
            -0.48154911416242535,
            -0.3524127792488796,
            -0.19267976678027168,
            -0.03661655847054497,
            0.09217776155776454,
            0.17874859040445815,
            0.23317374533305077,
            0.2730714837685855
        ),
        (
            0.9999999999978245,
            0.86920594512484,
            0.7279448382698354,
            0.5599201011313113,
            0.35480090159207917,
            0.1251343747071331,
            -0.11578482657191339,
            -0.3386099879667417,
            -0.5237057605802977,
            -0.6493796743476659,
            -0.6896631759695764,
            -0.6228595637066056,
            -0.46493329522714855,
            -0.2668545027558236,
            -0.07448802353593012,
            0.10961271409742139
        ),
        (
            -0.9999999999979908,
            -0.5764819016240738,
            -0.16718480960538615,
            0.20682393900720708,
            0.5108727857673341,
            0.6884917924383599,
            0.7172321042040731,
            0.5819934269523048,
            0.30277097525637586,
            -0.05245855346655453,
            -0.37393926342154094,
            -0.5429484627089037,
            -0.5015104458883564,
            -0.27914679362245054,
            0.056267775828227375,
            0.4288490248964723
        )
    ),
    "0.01": (
        (
            -0.8626483863937229,
            -0.5421146226548403,
            -0.22828313821751003,
            0.06537570626477504,
            0.32242683539135286,
            0.5273737366079322,
            0.6615275803317069,
            0.7109611547733258,
            0.6697753944568755,
            0.5434662139580849,
            0.3475195724702349,
            0.10538359474252713,
            -0.16235644090876278,
            -0.4389303496495133,
            -0.7191441451512912,
            -0.9999999999977569
        ),
        (
            0.999999999998209,
            0.7747317718558482,
            0.5502055699667029,
            0.32917768685356324,
            0.11659316297281415,
            -0.0799789688166225,
            -0.2497821251930224,
            -0.3809591092095561,
            -0.4644767952373422,
            -0.4948471287648065,
            -0.47092099474368804,
            -0.3965347513184376,
            -0.2814707628458812,
            -0.13994134813371337,
            0.014608722244616514,
            0.17384837543234824
        ),
        (
            -0.9999999999980417,
            -0.5673401976501716,
            -0.15213419491989671,
            0.2186771646543816,
            0.509079856820157,
            0.6737843502450064,
            0.6913228123444268,
            0.5540367336508464,
            0.2872220322393181,
            -0.044142826680927054,
            -0.3449647967440609,
            -0.5099586869627786,
            -0.4850629020189032,
            -0.2800589956498401,
            0.040768425216911465,
            0.40695599590031084
        )
    )
}


# Why a *trainable* mixer rather than any of these projections fixed in place.
#
# lda3 was measured at 0.2810 mAP against pseudo_rgb's 0.4291 under an identical
# recipe, despite 3.1x the material separability. The reason is not lost edge
# structure -- the projected frame has *more* gradient than pseudo_rgb -- it is
# that the extra gradient is noise. Measuring how much of each frame's gradient
# survives a 3x3 blur, which noise does not and structure does:
#
#   variant                 gradient   edge fraction
#   pseudo_rgb                 5.938           0.550
#   lda3, penalty 0            8.212           0.316
#   lda3, penalty 0.001        7.336           0.353
#   lda3, penalty 0.01         7.376           0.375
#
# A projection is a weighted *difference* of bands, and differences amplify
# noise where a single band does not. Smoothing the weights helps but cannot
# close the gap, and Savitzky-Golay smoothing of the cube beforehand does not
# either (0.316 -> 0.327).
#
# This is the argument for the adapter. A fixed projection is fitted for
# separability alone and has no way to know it is manufacturing noise; a mixer
# trained against the detection loss trades separability for signal-to-noise on
# its own, because noise costs it detections. The projections here are its
# starting prior, not its answer.


def gaussian_srf_bank(n_bands: int = 16, k: int = 8, width: float = 2.0):
    """A bank of k non-negative spectral response curves over n_bands.

    Each row is a Gaussian over band index, normalised to sum to one, so the
    reduction it performs is a weighted *average* of neighbouring bands. That is
    the property that matters: an average has variance ~1/n, while the signed
    projections above are differences whose weights cancel exactly
    (|sum(w)| / sum(|w|) is 0.000 for all of them) and which therefore amplify
    noise. Measured on rendered frames, the edge fraction -- how much of a
    frame's gradient survives a 3x3 blur, so structure rather than noise --
    separates the two families cleanly:

        pseudo_rgb (what the COCO stem was trained on)   0.563
        signed projection, unsmoothed                    0.324
        signed projection, smoothed                      0.390
        box SRF over three contiguous groups             0.561
        gaussian SRF, width 3                            0.572

    Centres are spread evenly across the spectrum and the curves overlap, as a
    real sensor's response functions do, so no band falls between two filters.
    """
    import numpy as _np

    if k < 1 or n_bands < 1:
        raise ValueError(f"need k >= 1 and n_bands >= 1, got k={k}, n_bands={n_bands}")
    idx = _np.arange(n_bands, dtype=_np.float64)
    # Centres sit inside the range rather than on its edges, so the outermost
    # filters still have most of their mass over real bands.
    centres = _np.linspace(0, n_bands - 1, k + 2)[1:-1] if k > 1 else \
        _np.array([(n_bands - 1) / 2])
    bank = _np.stack([_np.exp(-0.5 * ((idx - c) / max(width, 1e-6)) ** 2)
                      for c in centres])
    return bank / bank.sum(axis=1, keepdims=True)


def band_group_mixing(n_bands: int = 16, out_ch: int = 3):
    """``out_ch`` contiguous band-group averages: what an RGB sensor does.

    Rows are non-negative and sum to one, and the groups partition the spectrum
    in order, so the three outputs stand in the same long/mid/short relationship
    to each other that R, G and B do. That is the target the mixer starts from.
    """
    import numpy as _np

    edges = [round(i * n_bands / out_ch) for i in range(out_ch + 1)]
    T = _np.zeros((out_ch, n_bands))
    for c in range(out_ch):
        lo, hi = edges[c], edges[c + 1]
        T[c, lo:hi] = 1.0 / (hi - lo)
    return T


def srf_to_rgb_init(bank, out_ch: int = 3):
    """Least-squares ``out_ch`` x k mixer taking an SRF bank to band groups.

    The mixer is trainable, so this only has to start it somewhere the
    pretrained stem can already read. Composing it with the bank reproduces
    :func:`band_group_mixing` to within 0.061 per element at k=8, and the frame
    it renders measures at the same edge fraction as the composite the COCO
    stem was trained on (0.759 against pseudo_rgb's 0.744 over 40 frames, on the
    same metric that puts the signed discriminant at 0.537).
    """
    import numpy as _np

    bank = _np.asarray(bank, dtype=_np.float64)
    return band_group_mixing(bank.shape[1], out_ch) @ _np.linalg.pinv(bank)
