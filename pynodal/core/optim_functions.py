import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from pupil_masks import *
import os
#import poppy
import tqdm

from matplotlib.colors import LogNorm



def myfft2(wave_in):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave_in)))


def myifft2(wave_in):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wave_in)))


def cart2pol(x, y):
    """
    Takes cartesian (2D) coordinates and transforms them into polar.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def PolarTransform(t, rstep, thetastep_deg):
    N = np.shape(t)[0]
    center = N / 2 + 1
    thetastep = thetastep_deg * np.pi / 180
    polartrans = np.zeros((int(N / 2 / rstep), int(360 / thetastep_deg)), dtype=np.complex)

    rs = np.linspace(0, N / 2, int(N / 2 / rstep), endpoint=False)
    qs = np.linspace(0, 2 * np.pi, int(2 * np.pi / thetastep), endpoint=False)
    Rs, Qs = np.meshgrid(rs, qs)

    Ylocs = center + Rs * np.sin(Qs)
    Xlocs = center + Rs * np.cos(Qs)
    Xlocs[Xlocs == N] = N - 1
    Ylocs[Ylocs == N] = N - 1
    for i in np.arange(0, len(rs)):
        for j in np.arange(0, len(qs)):
            polartrans[i, j] = t[Ylocs[j, i], Xlocs[j, i]]

    return polartrans


def iPolarTransform(im, rstep_pix, thetastep_deg):
    num_radii, num_angles = np.shape(im)
    N = int(2 * num_radii * rstep_pix)
    ipolartrans_real = np.zeros((N, N), dtype=np.complex)
    ipolartrans_imag = np.zeros((N, N), dtype=np.complex)
    ix = np.arange(-N / 2, N / 2)
    X, Y = np.meshgrid(ix, ix)

    R_i = np.round(np.sqrt(X ** 2 + Y ** 2) / rstep_pix) + 1
    Q_i = np.round((np.arctan(Y, X) + np.pi) * 180 / np.pi / thetastep_deg) + 1
    Q_i[Q_i == num_angles] = 1
    R_i[R_i >= num_radii] = 1

    in_real = np.real(im)
    in_imag = np.imag(im)
    for k in np.arange(0, len(ix)):
        for p in np.arange(0, len(ix)):
            ipolartrans_real[k, p] = in_real[R_i[k, p], Q_i[k, p]]
            ipolartrans_imag[k, p] = in_imag[R_i[k, p], Q_i[k, p]]

    ipolartrans = ipolartrans_real + 1j * ipolartrans_imag
    ipolartrans[R_i > num_radii] = 0
    ipolartrans = np.rot90(ipolartrans, 2)
    return ipolartrans


def downsample_mask(path, mask_name):
    filename = os.path.join(path, mask_name)
    filename += '.fits'
    fpm = fits.getdata(filename) / np.pi
    h = fits.getheader(filename)
    mask_8bit = (fpm + 1) * 256 / 2
    mask_8bit = mask_8bit.astype(np.uint8)
    hdu = fits.PrimaryHDU(data=mask_8bit)
    savename = os.path.join(path, mask_name) + '_8bit.fits'
    hdu.header = h
    hdu.writeto(savename)
    return mask_8bit


def plot_im_ld(image, ld, w, n=2, color='viridis', log_scale=False, v_min=0, v_max=1, log_max=0, log_min=-3):
    l = np.shape(image)[0]
    n0 = int(l / 2)
    width = int(w * ld)
    crop = image[n0 - width:n0 + width, n0 - width:n0 + width]
    norm = None
    if log_scale:
        v_min = 10 ** log_min
        v_max = 10 ** log_max
        norm = LogNorm()

    n1 = int(np.shape(crop)[0] / 2)

    ax = plt.imshow(crop, interpolation='nearest', cmap=color, norm=norm, vmin=v_min, vmax=v_max)
    labs = np.arange(-w, w + 2, 2)
    locs = np.arange(0, 2 * w * ld + ld, 2 * ld)
    plt.xticks(locs, [str(abs(lab)) for lab in labs])
    plt.yticks(locs[:-1], [str(abs(lab)) for lab in labs[:-1]])
    if n is not None:
        circ = plt.Circle((n1, n1), radius=n * ld, color='red', linestyle='--', alpha=1, fill=False)
        plt.gca().add_patch(circ)
    plt.colorbar()
    return


def plot_im_LP(LP, apRad, w, color='gray', **kwargs):
    l = np.shape(LP)[0]
    n0 = int(l / 2)
    w = int(apRad * w + apRad)
    crop = LP[n0 - w:n0 + w, n0 - w:n0 + w]
    plt.imshow(crop, interpolation='nearest', cmap=color, **kwargs)
    plt.colorbar()
    return


def calculate_contrast_curve(PSF, lD):
    rad = 8 * lD
    flux = []
    for r in np.arange(0, 8 * lD):
        flux.append(np.mean(PSF[np.where(RHO == r)]))
    x = np.arange(len(flux)) / lD
    return x, flux


def throughputLP(LP, LS, EP):
    return np.sum((LP * LS) ** 2) / np.sum((EP * LS) ** 2)


def fqpm(THETA, N):
    FQPM = np.zeros((N, N))
    for k in np.arange(-2, 2):
        FQPM[np.where((THETA < (k + 1) * np.pi / 2) & (THETA >= k * np.pi / 2))] = (-1) ** k

    FQPM[FQPM < 0] = 0
    FQPM *= np.pi
    return np.exp(1j * FQPM)


def calcMask(EP, FPMinit, LS, OPT_REG_FP, OPT_REG_LP, max_its, goal_leak, ROI=False, ROIval=None, lD=10):
    """
    Calculates an optimized mask starting from a vortex, for a given entrance pupil EP
    :param EP: Entrance pupil given on a NxN grid axis 0,0
    :param FPMinit: initial focal plane mask
    :param LS: Lyot stop mask
    :param OPT_REG_FP:
    :param OPT_REG_LP:
    :param max_its: max number of iterations
    :goal_leak: fraction of leaked light desirable in the end
    """
    if ROIval is None:
        ROIval = [1.5, 2.5]
    PSF = myfft2(EP)  # field at focal plane
    FP = PSF * FPMinit  # field just after initial focal plane mask

    count = 0  # counter for iterations
    leaks = []  # array to store leak at each iteration
    leak = 1  # intialize leak to 1
    maxleak = np.sum(np.sum(abs(EP * LS) ** 2))
    if ROI:
        xx, yy = np.meshgrid(np.arange(-512, 512), np.arange(-512, 512))
        Rho, Theta = cart2pol(xx, yy)
        OPT_REG_FP_off = np.zeros((1024, 1024))
        OPT_REG_FP_off[np.where(Rho < ROIval[1] * lD)] = 1
        OPT_REG_FP_off[np.where(Rho < ROIval[0] * lD)] = 0
        OPT_REG_FP_off = np.lib.pad(OPT_REG_FP_off, (1536, 1536), 'constant', constant_values=(0, 0))

    for count in tqdm.tnrange(max_its):  # and leak > goal_leak
        # count += 1
        LPj = myifft2(FP)  # field in Lyot plane
        LPj *= (1 - OPT_REG_LP)  # field in LP with nodal region set to zero
        FPj = myfft2(LPj)  # current focal plane
        FPC = np.exp(1j * OPT_REG_FP * (np.angle(FPj) - np.angle(PSF) - np.angle(FPMinit)))

        leak = np.sum(np.sum(np.real(LPj * LS)) ** 2) / maxleak
        FP = PSF * FPMinit * FPC  # current phase mask, updated focal plane
        if ROI and (count % 10 == 0):
            FP[OPT_REG_FP_off == 1] = 0
        leaks.append(leak)

    del FPj
    del LPj
    its = count  # final count
    FPM = np.exp(1j * (np.angle(FP) - np.angle(PSF)))
    FPC = np.exp(1j * (np.angle(FP) - np.angle(PSF) - np.angle(FPMinit)))
    plt.plot(np.arange(len(leaks)), leaks)

    finalleak = leak
    return FPM, FPC, its, finalleak


def modified_GS_amplitude(EP, optEPinit, OPT_REG_FP, max_its):
    PSF = myfft2(EP)
    count = 0
    init_amplitude = np.sum(np.sum(PSF * OPT_REG_FP) ** 2)
    amplitudes = []
    EPj = optEPinit
    while count < max_its:
        count += 1
        PSFj = myfft2(EP * EPj)
        a = np.sum(np.sum(np.real(PSFj * OPT_REG_FP)) ** 2) / init_amplitude
        amplitudes.append(a)
        PSFj *= (1 - OPT_REG_FP)
        CEPj = myifft2(PSFj)
        CEPj = abs(CEPj)
        EPj = CEPj + (1 - EP)

    plt.semilogy(np.arange(len(amplitudes)), amplitudes, 'o')
    return EPj, amplitudes[-1]


def modified_GS_phase(EP, optEPinit, OPT_REG_FP, max_its):
    PSF = myfft2(EP)
    count = 0
    init_amplitude = np.sum(np.abs(PSF) * OPT_REG_FP)
    amplitudes = []
    EPj = np.exp(1j * optEPinit)
    for count in tqdm.tqdm(range(max_its)):
        PSFj = myfft2(EP * EPj)
        a = np.sum(np.abs(PSFj) * OPT_REG_FP) / init_amplitude
        amplitudes.append(a)
        PSFj *= (1 - OPT_REG_FP)
        CEPj = myifft2(PSFj)
        EPj = np.exp(1j * np.angle(CEPj))
    plt.semilogy(np.arange(len(amplitudes)), amplitudes, 'o')
    print(amplitudes[-1])
    return EPj


def _nHexesInRing(n):
    """ How many hexagons in ring N? """
    return 1 if n==0 else 6*n

def nHexesInsideRing(n):
    """ How many hexagons interior to ring N, not counting N?"""
    return sum( [_nHexesInRing(i) for i in range(n)])


# with circular symmetry enforced
def calcMask_forcesymmetry(EP, FPMinit, LS, OPT_REG_FP, OPT_REG_LP, max_its, goal_leak):
    """
    Calculates an optimized mask starting from a vortex, for a given entrance pupil EP
    :param EP: Entrance pupil given on a NxN grid axis 0,0
    :param FPMinit: initial focal plane mask
    :param LS: Lyot stop mask
    :param OPT_REG_FP:
    :param OPT_REG_LP:
    :param max_its: max number of iterations
    :goal_leak: fraction of leaked light desirable in the end
    """
    every = 10

    PSF = myfft2(EP)  # field at focal plane
    FP = PSF * FPMinit  # field just after initial focal plane mask

    count = 0  # counter for iterations
    leaks = []  # array to store leak at each iteration
    leak2 = 1  # intialize leak to 1
    maxleak = np.sum(np.sum(abs(EP * LS) ** 2))

    while (count < max_its and leak2 > goal_leak):
        count += 1
        LPj = myifft2(FP)  # field in Lyot plane
        LPj *= (1 - OPT_REG_LP)  # field in LP with nodal region set to zero
        FPj = myfft2(LPj)  # current focal plane

        FPC0 = np.exp(1j * (np.angle(FPj) - np.angle(PSF) - np.angle(FPMinit)))
        leak = np.sum(np.sum(np.real(LPj * LS)) ** 2) / maxleak

        if count % every == 0:
            pc_FPC = PolarTransform(FPC0, 1, 1)
            ftpc_FPC = np.fft.fftshift(np.fft.fft(np.fft.fftshift(pc_FPC, axes=(1,)), axis=1), axes=(1,))
            ftpc_FPC_new = np.zeros(np.shape(ftpc_FPC), dtype=np.complex)

            ftpc_FPC_new[:, 180] = ftpc_FPC[:, 180]

            pc_FPC = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(ftpc_FPC_new, axes=(1,)), axis=1), axes=(1,))
            FPC0 = iPolarTransform(pc_FPC, 1, 1)

        if count % every == 0:
            leak2 = np.sum(np.sum(abs(LPj * LS) ** 2)) / maxleak

        FPC = np.exp(1j * OPT_REG_FP * np.angle(FPC0))

        FP = PSF * FPMinit * FPC  # current phase mask, updated focal plane
        leaks.append(leak)

    del FPj
    del LPj
    its = count  # final count
    FPM = np.exp(1j * (np.angle(FP) - np.angle(PSF)))
    FPC = np.exp(1j * (np.angle(FP) - np.angle(PSF) - np.angle(FPMinit)))
    finalleak = leak
    plt.plot(np.arange(len(leaks)), leaks)
    return FPM, FPC, its, finalleak