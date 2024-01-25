"""Main module."""


import numpy as np
import spdist
import scipy.optimize as opt
from scipy.optimize import least_squares

import matplotlib.pyplot as plt


def _residue(
    p: list,
    energy_grid: np.ndarray,
    spectrum_x: np.ndarray,
    spectrum_y: np.ndarray,
    ref_spectrum_x: np.ndarray,
    ref_spectrum_y: np.ndarray,
    fit_range: list | None = None,
):
    """Residue to calculate the shift and scale of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric

    Args:
        p(list): shift and scale
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        spectrum_x(np.ndarray): energy grid of the spectrum
        spectrum_y(np.ndarray): mu of the spectrum
        ref_spectrum_x(np.ndarray): energy grid of the reference spectrum
        ref_spectrum_y(np.ndarray): mu of the reference spectrum

    Returns:
        residue(np.ndarray): residue of the spectrum and the reference spectrum
    """

    # print(len(spectrum_y))
    spectrum_y = np.interp(energy_grid, spectrum_x + p[0], spectrum_y * p[1])

    if fit_range:
        index = np.where((energy_grid >= fit_range[0]) & (energy_grid <= fit_range[1]))
        energy_grid = energy_grid[index]
        spectrum_y = spectrum_y[index]

    # print(len(spectrum_y))
    # print(spdist.spdist(energy_grid, spectrum_y, ref_spectrum_x, ref_spectrum_y))
    # print(spdist.spdist(energy_grid, spectrum_y, ref_spectrum_x, ref_spectrum_y))
    # print(
    #     spdist.squared_spdist(energy_grid, spectrum_y, ref_spectrum_x, ref_spectrum_y)
    # )
    return spdist.spdist(
        energy_grid, spectrum_y, ref_spectrum_x, ref_spectrum_y
    ) + np.mean(
        np.abs(spectrum_y - np.interp(energy_grid, ref_spectrum_x, ref_spectrum_y))
    )


def _residue_MSE(
    p: list,
    energy_grid: np.ndarray,
    spectrum_x: np.ndarray,
    spectrum_y: np.ndarray,
    ref_spectrum_x: np.ndarray,
    ref_spectrum_y: np.ndarray,
    fit_range: list | None = None,
):
    """Residue to calculate the shift and scale of the spectrum, with respect to the reference spectrum_using MSE as the metric

    Args:
        p(list): shift and scale
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        spectrum_x(np.ndarray): energy grid of the spectrum
        spectrum_y(np.ndarray): mu of the spectrum
        ref_spectrum_x(np.ndarray): energy grid of the reference spectrum
        ref_spectrum_y(np.ndarray): mu of the reference spectrum

    Returns:
        residue(np.ndarray): residue of the spectrum and the reference spectrum
    """

    # print(len(spectrum_y))
    spectrum_y = np.interp(energy_grid, spectrum_x + p[0], spectrum_y * p[1])

    if fit_range:
        index = np.where((energy_grid >= fit_range[0]) & (energy_grid <= fit_range[1]))
        energy_grid = energy_grid[index]
        spectrum_y = spectrum_y[index]

    return np.mean(
        (spectrum_y - np.interp(energy_grid, ref_spectrum_x, ref_spectrum_y)) ** 2
    )
    # return np.sum(
    #     (
    #         spectrum_y
    #         - np.interp(spectrum_x, ref_spectrum_x - p[0], ref_spectrum_y / p[1])
    #     )
    #     ** 2
    # )


def calc_shift_scale(
    energy_grid,
    spectrum_x,
    spectrum_y,
    ref_spectrum_x,
    ref_spectrum_y,
    fit_range=None,
    max_shift=20,
):
    """Calculate the shift and scale of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric

    Args:
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        spectrum_x(np.ndarray): energy grid of the spectrum
        spectrum_y(np.ndarray): mu of the spectrum
        ref_spectrum_x(np.ndarray): energy grid of the reference spectrum
        ref_spectrum_y(np.ndarray): mu of the reference spectrum
        fit_range(list): the fitting range for the metric calculation
        max_shift(float): the maximum shift of the spectrum

    Returns:
        shift(float): shift of the spectrum
        scale(float): scale of the spectrum
        loss(float): loss of the spectrum
    """

    if fit_range:
        index = np.where(
            (ref_spectrum_x >= fit_range[0] - max_shift)
            & (ref_spectrum_x <= fit_range[1] + max_shift)
        )
        ref_spectrum_x = ref_spectrum_x[index]
        ref_spectrum_y = ref_spectrum_y[index]

    p0 = [0, 1]

    residue = lambda p: _residue(
        p,
        energy_grid,
        spectrum_x,
        spectrum_y,
        ref_spectrum_x,
        ref_spectrum_y,
        fit_range,
    )

    optimization_algorithm = opt.shgo(residue, [(-20, 20), (0.5, 1.5)])

    shift, scale = optimization_algorithm.x
    loss = optimization_algorithm.fun
    print(loss)

    return shift, scale, loss


def calc_shift_scale_MSE(
    energy_grid,
    spectrum_x,
    spectrum_y,
    ref_spectrum_x,
    ref_spectrum_y,
    fit_range=None,
    max_shift=20,
):
    """Calculate the shift and scale of the spectrum, with respect to the reference spectrum using MSE as the metric

    Args:
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        spectrum_x(np.ndarray): energy grid of the spectrum
        spectrum_y(np.ndarray): mu of the spectrum
        ref_spectrum_x(np.ndarray): energy grid of the reference spectrum
        ref_spectrum_y(np.ndarray): mu of the reference spectrum
        fit_range(list): the fitting range for the metric calculation
        max_shift(float): the maximum shift of the spectrum

    Retruns:
        shift(float): shift of the spectrum
        scale(float): scale of the spectrum
        loss(float): loss of the spectrum
    """
    if fit_range:
        index = np.where(
            (ref_spectrum_x >= fit_range[0] - max_shift)
            & (ref_spectrum_x <= fit_range[1] + max_shift)
        )
        ref_spectrum_x = ref_spectrum_x[index]
        ref_spectrum_y = ref_spectrum_y[index]

    p0 = [0, 1]

    residue = lambda p: _residue_MSE(
        p,
        energy_grid,
        spectrum_x,
        spectrum_y,
        ref_spectrum_x,
        ref_spectrum_y,
        fit_range,
    )

    results = least_squares(residue, p0)

    shift, scale = results.x
    loss = results.cost

    return shift, scale, loss


def test():
    """Test function for the module"""
    exp_data = np.loadtxt("examples/Ptfoil.nor")
    theoretical_data = np.loadtxt("examples/Pt_foil/xmu.dat")

    ref_spectrum_x = exp_data[:, 0]
    ref_spectrum_y = exp_data[:, 1]

    spectrum_x = theoretical_data[:, 0]
    spectrum_y = theoretical_data[:, 3]

    spectrum_x = spectrum_x - 10
    spectrum_y = spectrum_y
    e0 = 11564
    fit_range = [e0 - 20, e0 + 80]
    energy_grid = np.linspace(fit_range[0], fit_range[1], 100)

    shift, scale, loss = calc_shift_scale(
        energy_grid,
        spectrum_x,
        spectrum_y,
        ref_spectrum_x,
        ref_spectrum_y,
        fit_range=fit_range,
        max_shift=20,
    )

    print(shift, scale)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(ref_spectrum_x, ref_spectrum_y, label="Pt foil(exp)")
    ax.plot(spectrum_x, spectrum_y, label="Pt foil(feff)")

    ax.plot(
        spectrum_x + shift,
        spectrum_y * scale,
        label="Pt foil(feff: scale spdist + MAE)",
    )

    shift, scale, loss = calc_shift_scale_MSE(
        energy_grid,
        spectrum_x,
        spectrum_y,
        ref_spectrum_x,
        ref_spectrum_y,
        fit_range=fit_range,
        max_shift=20,
    )

    ax.plot(spectrum_x + shift, spectrum_y * scale, label="Pt foil(feff: scale MSE)")

    ax.set_xlim(*fit_range)
    ax.legend()

    fig.savefig("./examples/Pt_foil_comparison.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    test()
