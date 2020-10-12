"""
Interpolate the solution of a linear system computed on a mesh to another mesh.
"""

import numpy as np

from fenics import *
from dolfin import *
from mshr import *

# Filepaths
data_path = "../../data/PDE_example/"
plot_path = "../plots/"

# Settings
parameters["reorder_dofs_serial"] = False  # same mesh and linear system order


# -- Load Mesh and create Function Space --
mesh_res_coarse = 6
mesh_res_fine = 128
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
np.save("{}solution_interpol_res{}tores{}".format(data_path, mesh_res_fine,
                                                  mesh_res_coarse),
        u_interpol_vec)
