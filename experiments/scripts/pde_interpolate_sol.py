"""
Interpolate the solution of a linear system computed on a mesh to another mesh.
"""

import argparse
import os
import sys

import numpy as np

from fenics import *
from dolfin import *
from mshr import *


def main(args):
    """
    Main entry point allowing external calls

    Parameters
    ----------
    args : list
        command line parameter list
    """
    args = parse_args(args)
    # Filepaths
    data_path = args.data_path

    # Settings
    parameters["reorder_dofs_serial"] = False  # same mesh and linear system order

    # Mesh sizes
    mesh_res_coarse = args.resolutions[0]
    mesh_res_fine = args.resolutions[1]

    # -- Load Mesh and create Function Space --
    mesh_coarse = Mesh(data_path + "mesh_res{}.xml".format(mesh_res_coarse))
    mesh_fine = Mesh(data_path + "mesh_res{}.xml".format(mesh_res_fine))

    V_coarse = FunctionSpace(mesh_coarse, "P", 1)
    V_fine = FunctionSpace(mesh_fine, "P", 1)

    # -- Load Solution --
    u_vec = np.load(file=data_path + "solution_res{}.npy".format(mesh_res_fine))
    u = Function(V_fine)
    u.vector()[:] = u_vec

    # -- Interpolate Solution --
    u.set_allow_extrapolation(True)  # fine and coarse mesh might not coincide
    u_interpol = interpolate(u, V_coarse)

    # -- Save Interpolation --
    u_interpol_vec = np.array(u_interpol.vector().get_local())
    np.save(
        "{}solution_interpol_res{}tores{}".format(
            data_path, mesh_res_fine, mesh_res_coarse
        ),
        u_interpol_vec,
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
    parser = argparse.ArgumentParser(
        description="Create mesh and linear system of a PDE via Galerkins method."
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="data_path",
        help="filepath to save data at",
        default="../../data/Galerkins_method/",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resolutions",
        dest="resolutions",
        help="Mesh resolutions.",
        default=[6, 128],
        type=list,
    )
    return parser.parse_args(args)


def run():
    """
    Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
