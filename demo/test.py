from bbidomain import cressman_FE, VectorDouble
import numpy as np
import time

N = 5
dt = 0.001
t0 = 0
t1 = 100000

V = VectorDouble(np.ones(N)*-50.0)
m = VectorDouble(np.ones(N)*0.0936)
n = VectorDouble(np.ones(N)*0.96859)
h = VectorDouble(np.ones(N)*0.08553)
Ca = VectorDouble(np.ones(N)*0.0)
K = VectorDouble(np.ones(N)*7.8)
Na = VectorDouble(np.ones(N)*15.5)

tick = time.perf_counter()
result = cressman_FE(V, m, n, h, Ca, K, Na, t0, t1, dt)
tock = time.perf_counter()
print(f"time: {tock - tick}")
# print(result)


import matplotlib.pyplot as plt
plt.plot(result)
plt.savefig("foo.png")
