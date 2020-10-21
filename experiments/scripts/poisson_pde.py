"""
Compute the linear system resulting from discretization of the PDE below:

Poisson equation with Dirichlet conditions.

  -Laplace(u) = f    in the interior
            u = u_D  on the boundary

where
          u_D = (x^2 - 2 * y)^2 * (1 + sin(2 * pi * x))
            f = 15
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

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

    # Mesh size
    resolution = args.resolution

    # -- Create Mesh --

    # Specify an ellipse geometry
    center = Point(0.0, 0.0)
    horizontal_semi_axis = 1.0
    vertical_semi_axis = 1.0
    e = Ellipse(center, horizontal_semi_axis, vertical_semi_axis)

    # Generate mesh with given resolution
    mesh = generate_mesh(e, resolution)

    # Save mesh
    mesh_file = File(data_path + "mesh_res{}.xml".format(resolution))
    mesh_file << mesh

    # Plot the mesh
    plot(mesh)
    plt.axis("off")
    # plt.show()

    # Define function space
    V = FunctionSpace(mesh, "P", 1)

    # Save mesh
    mesh_xy = mesh.coordinates()
    mesh_cells = mesh.cells()
    np.save(data_path + "mesh_xy_res{}".format(resolution), mesh_xy)
    np.save(data_path + "mesh_cells_res{}".format(resolution), mesh_cells)

    # -- Define Dirichlet Problem --

    # Define boundary condition
    u_D = Expression("pow(pow(x[0], 2) - 2*x[1], 2)*(1 + sin(2 * pi * x[0]))", degree=4)

    def boundary(x, on_boundary):
        """Boundary condition"""
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(15.0)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    # -- Assemble linear system --

    # Assemble system by applying boundary conditions
    M_A, v_b = assemble_system(a, L, bc)  # preserves symmetry

    # Matrix (discretized differential operator)
    A = scipy.sparse.csr_matrix(np.array(M_A.array()))

    # Right hand side
    b = np.array(v_b.get_local())

    # Solve using CG
    # unp = scipy.sparse.linalg.spsolve(A, b)

    # Save linear system to file
    scipy.sparse.save_npz("{}matrix_poisson_res{}.npz".format(data_path, resolution), A)
    np.save("{}rhs_poisson_res{}".format(data_path, resolution), b)

    # -- Solution --

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution incl. mesh
    plot(u)
    plot(mesh, color="white", alpha=0.3)
    plt.show()


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
        "--resolution",
        dest="resolution",
        help="Mesh resolution.",
        default=6,
        type=int,
    )
    return parser.parse_args(args)


def run():
    """
    Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
