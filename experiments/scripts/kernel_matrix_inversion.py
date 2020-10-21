# -*- coding: utf-8 -*-
"""
Uncertainty calibration for kernel Gram matrix inversion.

This script generates Gram matrices K from a given kernel k and solves linear systems of the form :math:`K x = b` using
a probabilistic linear solver with and without prior information about the spectrum. The posterior uncertainty
calibration is then measured and compared.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing
import GPy
import probnum

_logger = logging.getLogger(__name__)


def load_dataset(file):
    """
    Load data from file.

    Parameters
    ----------
    file : str
        Filepath to the data in csv format.

    Returns
    -------
    data : np.ndarray, shape=(n,p)
    """
    # Load csv
    df = pd.read_csv(
        filepath_or_buffer=file, compression="zip", header=0, sep=",", quotechar='"'
    )

    # Drop unnamed columns and remove NaNs
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna()
    X = df.to_numpy(dtype=float)

    # Scale data
    X_scaled = preprocessing.scale(X=X, axis=0, with_mean=True, with_std=True)

    return X_scaled


def generate_linear_system(data, kernel, n_datapoints, sigma=10 ** -6):
    """
    Generate linear system with a kernel system matrix.

    Parameters
    ----------
    data : np.ndarray, shape=(n,p)
        Data matrix containing rows of feature vectors.
    kernel : str
        String describing kernel to apply to dataset. One of `["rbf", "matern32", "matern52"]`
    n_datapoints : int
        Number of datapoints to sample.
    sigma : float
        Scalar for the regularization term added to the kernel matrix to ensure positive definiteness.

    Returns
    -------
    x_true : np.ndarray
        Solution to the linear system.
    kern_mat : np.ndarray
        Kernel Gram matrix.
    rhs : np.ndarray
        Right-hand-side of the linear system.
    """
    # Sample data from dataset
    idx = np.random.choice(data.shape[0], n_datapoints, replace=False)
    X = data[idx, :]

    # Compute kernel Gram matrix
    kernel = kernel.lower()
    if kernel == "rbf":
        kern = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    elif kernel == "matern32":
        kern = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=1)
    elif kernel == "matern52":
        kern = GPy.kern.Matern52(input_dim=1, variance=1, lengthscale=1)
    else:
        raise ValueError("Chosen kernel not recognized.")
    kern_mat = kern.K(X=X)
    kern_mat = kern_mat + sigma * np.eye(kern_mat.shape[0])
    print(f"Matrix condition of {kernel} kernel matrix: {np.linalg.cond(kern_mat)}")

    # Sample random solution
    x_true = np.random.normal(size=(n_datapoints, 1))

    # Compute right hand side
    rhs = kern_mat @ x_true

    return x_true, kern_mat, rhs


def compute_test_statistic(x_true, x_est, trace_solution_cov):
    """
    Compute calibration test statistic.

    Computes the calibration statistic w = 1/2 ln(tr(Cov(x))) - ln(||x_* - E[x]||_2). The linear solver is overconfident
    for w < 0, perfectly calibrated for w = 0 and underconfident w > 0.

    Parameters
    ----------
    x_true : np.ndarray
        Solution to the linear system :math:`Ax=b`.
    x_est : np.ndarray
        Estimated solution to the linear system :math:`Ax=b`.
    trace_solution_cov : float
        Trace :math:`\\tr(\\operatorname{Cov}(x))` of the solution covariance.

    Returns
    -------
    statistic: float
        Value of the test statistic.
    """
    error = x_true.ravel() - x_est.ravel()
    error_2_norm = np.linalg.norm(error, ord=2)
    return 0.5 * np.log(np.maximum(trace_solution_cov, 10 ** -16)) - np.log(
        error_2_norm
    )


def main(args):
    """
    Main entry point allowing external calls

    Parameters
    ----------
    args : list
        command line parameter list
    """
    # Setup
    args = parse_args(args)
    setup_logging(args.loglevel)
    np.random.seed(seed=args.seed)
    if args.n_samples is None:
        n_samples = (10 ** 5 / np.array(args.datapoints)).astype(int)
    else:
        n_samples = np.repeat(args.n_samples, len(args.datapoints))

    # Load data
    _logger.debug("Loading dataset.")
    data = load_dataset(file=args.file_data)

    # Output table
    df_teststats = pd.DataFrame(
        columns=["kernel", "dimension", "n_iters", "calibrated", "statistic", "sample"]
    )
    df_stat_dist = pd.DataFrame(
        columns=["kernel", "dimension", "calib_method", "stat_avg", "stat_std"]
    )

    # Sample data and compute Gram matrices for a given kernel and number of data points
    _logger.debug("Solving linear systems.")
    for kernel in args.kernels:
        for i, d in enumerate(args.datapoints):
            # Setup
            test_stats_tmp = np.empty(shape=(n_samples[i], len(args.calib_methods)))

            for n in range(n_samples[i]):

                # Sample linear problem
                print(f"Sample linear problem {n + 1}/{n_samples[i]}.")
                sigma = 10 ** -6 * d
                x_true, kernel_mat, b = generate_linear_system(
                    data=data, kernel=kernel, n_datapoints=d, sigma=sigma
                )

                # Compute spectrum
                k_iter = 0
                print(f"Compute spectrum.")
                eigvals = np.real_if_close(np.linalg.eigvals(kernel_mat))

                for i_calib_method, calib_method in enumerate(args.calib_methods):

                    # Calibration method
                    if calib_method == "none":
                        calib_mode = None
                    elif calib_method == "gpkern" or calib_method == "weightedmean":
                        calib_mode = calib_method
                    elif calib_method == "noise":
                        calib_mode = sigma
                    elif calib_method == "spectrum" and eigvals is not None:
                        calib_mode = np.real_if_close(np.mean(eigvals[k_iter::]))
                    else:
                        continue

                    # Solve linear system with PLS without calibration
                    print("Solve linear system.")
                    try:
                        xhat, _, _, info = probnum.linalg.problinsolve(
                            A=kernel_mat, b=b, calibration=calib_mode
                        )
                        k_iter = info["iter"]
                        print(info)

                        # Compute test statistic testing for calibration
                        test_statistic = compute_test_statistic(
                            x_true=x_true,
                            x_est=xhat.mean,
                            trace_solution_cov=info["trace_sol_cov"],
                        )
                    except np.linalg.LinAlgError:
                        test_statistic = np.nan
                        k_iter = 0

                    test_stats_tmp[n, i_calib_method] = test_statistic

                    # Append to dataframe
                    to_append = [kernel, d, k_iter, calib_method, test_statistic, n]
                    df_teststats = df_teststats.append(
                        pd.Series(to_append, index=df_teststats.columns),
                        ignore_index=True,
                    )

            # Compute statistical distance
            avg_test_statistics = np.nanmean(test_stats_tmp, axis=0)
            std_test_statistics = np.nanstd(test_stats_tmp, axis=0)

            # Append to dataframe
            for i_calib_method, calib_method in enumerate(args.calib_methods):
                to_append = [
                    kernel,
                    d,
                    calib_method,
                    avg_test_statistics[i_calib_method],
                    std_test_statistics[i_calib_method],
                ]
                df_stat_dist = df_stat_dist.append(
                    pd.Series(to_append, index=df_stat_dist.columns), ignore_index=True
                )

        # Write result to file
        _logger.debug("Writing result to file.")
        Path(args.file_out).mkdir(parents=True, exist_ok=True)
        df_teststats.to_csv(
            path_or_buf=os.path.join(args.file_out, "kernel_mat_inv_teststats.csv")
        )
        df_stat_dist.to_csv(
            path_or_buf=os.path.join(args.file_out, "kernel_mat_inv_table.csv")
        )


def setup_logging(loglevel):
    """
    Setup basic logging

    Parameters
    ----------
    loglevel : int
        minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args):
    """
    Parse command line parameters

    Parameters
    ----------
    args : list
        command line parameters as list of strings

    Returns
    -------
    argparse.Namespace : obj
        command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Kernel matrix inversion experiment.")
    parser.add_argument(
        "-f",
        "--file",
        dest="file_data",
        help="filepath to the data",
        default="../../data/kernel_matrix_inversion/flight_delay_jan2020.zip",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="file_out",
        help="output filepath",
        default="../tables/",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--kernels",
        dest="kernels",
        help="kernel functions of Gram matrices",
        default=["matern32", "matern52", "rbf"],
        type=list,
    )
    parser.add_argument(
        "-cm",
        "--calib_methods",
        dest="calib_methods",
        help="calibration method to use",
        default=["none", "weightedmean", "gpkern", "noise", "spectrum"],
        type=list,
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        dest="n_samples",
        help="number of linear systems to draw",
        # default=10,
        type=list,
    )
    parser.add_argument(
        "-d",
        "--datapoints",
        dest="datapoints",
        help="list of number of datapoints",
        default=[100, 1000, 10000],
        type=list,
    )
    parser.add_argument(
        "-s", "--seed", dest="seed", default=0, help="random seed", type=int
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def run():
    """
    Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
