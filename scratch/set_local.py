from dolfin import *
import numpy as np


mesh = UnitCubeMesh(10, 10, 10)
V = FunctionSpace(mesh, "CG", 1)

v = Function(V)

v_vec = v.vector()
local = v_vec.get_local()
local[:] = np.random.random(local.size)

v_vec.set_local(local)


comm = MPI.comm_world
rank = MPI.rank(comm)

print(f"Success!, {rank}")
