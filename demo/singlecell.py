import numpy as np
import matplotlib.pyplot as plt
from cressmanODE import solve_cressman_interval
import dolfin as df

# import ipdb

from xalbrain import BetterSingleCellSolver
from xalbrain.cellmodels import Cressman


def reference_solution(*, t0, t1):
    sol = solve_cressman_interval(t0, t1)
    # print(sol.y[:, -1])
    return sol.t, sol.y[0]


def better_solution(*, t0, t1):
    time = df.Constant(0)
    model = Cressman()
    solver = BetterSingleCellSolver(model=model, time=time, reload_ext_modules=False)
    vs_, *_ = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    vlist = [vs_.vector().get_local()[0]]
    tlist = [0]

    for interval, vs in solver.solve((t0, t1), dt=0.025):
        vlist.append(vs.vector().get_local()[0])
        tlist.append(interval[1])
        # print(vs.vector().get_local()[:7])
        # break
    return np.asarray(tlist), np.asarray(vlist)


if __name__ == "__main__":
    T0 = 0
    T1 = 10
    # T1 = 0.025

    tref, yref = reference_solution(t0=T0, t1=T1)
    print("Better Solver")
    tbetter, ybetter = better_solution(t0=T0, t1=T1)

    fig, ax = plt.subplots(1, constrained_layout=True)
    ax.plot(tref, yref, label="reference")
    ax.plot(tbetter, ybetter, label="better")
    ax.legend()
    fig.savefig("foo.png")
