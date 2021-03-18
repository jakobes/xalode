import numpy as np
import matplotlib.pyplot as plt


def rhs_fhn(y, t, args):
    c1, v_amp, v_rest, v_th, v_peak, c2, b, c3 = args
    v, w = y
    dv = c1/(v_amp**2)*(v - v_rest)*(v - v_th)*(v_peak - v) - c2*w
    # if t > 0.05 and t < 0.0625:
    #     dv += 5000
    dv += 500

    dw = b*(v - v_rest - c3*w)
    return dv, dw


def rhs_ml(y, t, args):
    Cm, Iext, gl, Vl, gca, VCa, gk, Vk, V1, V2, V3, V4, phi  = args
    Minf = (1 + np.tanh((y[0] - V1)/V2))/2
    Ninf = (1 + np.tanh((y[0] - V3)/V4))/2
    tau = phi*np.cosh((y[0] - V3)/(2*V4))
    dv = Cm*(Iext - gl*(y[0] - Vl) - gca*Minf*(y[0] - VCa) - gk*y[1]*(y[0] - Vk))
    dN = tau*(Ninf - y[1])
    return dv, dN


def fw(rhs, y0, t0, t1, dt, args):
    y = []
    t = []
    t.append(t0)
    y.append(y0)
    while t[-1] < t1:
        dv, dw = rhs(y[-1], t[-1], args)
        y.append([y[-1][0] + dt*dv, y[-1][1] + dt*dw])
        t.append(t[-1] + dt)
    return np.asarray(t), np.asarray(y)


def run_ml(Iext=40):
    V1 = -1.2
    V2 = 18.0
    V3 = 10.0
    V4 = 17.4
    phi = 1.0/15
    gl = 2.0
    gca = 4.0
    gk = 8.0
    Cm = 0.2
    Vl = -60
    VCa = 120
    Vk = -80

    args = Cm, Iext, gl,  Vl, gca, VCa, gk, Vk, V1, V2, V3, V4, phi
    t, y = fw(rhs_ml, [-60, 0], 0, 1000, 0.001, args)
    fig, ax = plt.subplots(1)
    ax.plot(t, y[:, 0], t, y[:, 1])
    fig.savefig("foo.png")



def run_fhn():
    a = 0.13            # 0.13
    b = 13.0            # 13.0
    c1 = 0.26           # 260
    c2 = 10.0           # 10
    c3 = 1.0            # 1.0
    v_rest = -65         # -0.07 V
    v_peak = 60         # 0.04 V
    v_amp = v_peak - v_rest
    v_th = v_rest + a*v_amp
    args = c1, v_amp, v_rest, v_th, v_peak, c2, b, c3

    y0 = [v_rest, 0]
    y0 = [v_th + 1, 0]
    t, y = fw(rhs_fhn, y0, 0, 1.0, 0.001, args)

    fig, ax = plt.subplots(1)
    ax.plot(t, y[:, 0], t, y[:, 1])
    fig.savefig("foo.png")


if __name__ == "__main__":
    run_ml()
