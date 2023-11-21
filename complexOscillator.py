import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

w0 = 2
gam = 0.5

def oscillator(t, y):
    '''y = [x, xDot]'''

    rate0 = y[1]
    rate1 = -(w0**2)*y[0] - gam*y[1]

    return [rate0, rate1]

Time = 10
dt = 0.01
sol = solve_ivp(oscillator, t_span=(0, Time), y0=[1, w0*1j], t_eval=np.arange(0, Time, dt))

T = sol.t
X = np.real(sol.y[0])
Y = np.imag(sol.y[0])

normalise = 5
XDot = np.real(sol.y[1])/normalise
yDot = np.imag(sol.y[1])/normalise

plt.style.use('dark_background')
fig, (ax, ax2) =  plt.subplots(1, 2, figsize=(20, 20))

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
""" ax.set_xlabel(r"Re $Z(t)$")
ax.set_ylabel(r"Im $Z(t)$") """
ax.set_aspect('equal')
ax.grid(color='#1f2a3d', linestyle='-.')

trace, = ax.plot([], [], '--', color="#FFFF00", lw=1)
arrow, = ax.plot([], [], 'o-', lw=2, color='#FF00FF', label=r"$\vec{E}$")
xTrace, = ax.plot([], [], '.-', color="#FF0000", label=r"$E_y$")
yTrace, = ax.plot([], [], '.-', color="#0000FF", label=r"$E_x$")
ax.legend(loc="upper right")

ax2.set_xlim(0, Time)
ax2.set_ylim(-1, 1)
ax2.set_xlabel("depth")
#ax2.set_ylabel(r"$y(t) =$ Im $z(t)$")
ax2.grid(color='#1f2a3d', linestyle='-.')

line, = ax2.plot([], [], color='#FF0000', label=r"$E_y$")
line2, = ax2.plot([], [], color='#0000FF', label=r"$E_x$")
exp, = ax2.plot([], [], '--', color="#FFFF00")
exp2, = ax2.plot([], [], '--', color="#FFFF00")
ax2.legend(loc="upper right")

def animate(frame):
    retain = 0

    trace.set_data(X[:frame][-retain:], Y[:frame][-retain:])
    arrow.set_data([0, X[frame]], [0, Y[frame]])
    
    xTrace.set_data([X[frame]]*10, np.linspace(0, Y[frame], 10))
    yTrace.set_data(np.linspace(0, X[frame], 10), [Y[frame]]*10)

    line.set_data(T[:frame], Y[:frame])
    line2.set_data(T[:frame], X[:frame])
    exp.set_data(T[:frame], np.exp(-gam/2 * T[:frame]))
    exp2.set_data(T[:frame], -np.exp(-gam/2 * T[:frame]))

    return trace, xTrace, yTrace, line, arrow, line2, exp, exp2

ani = animation.FuncAnimation(fig, animate, len(T), interval=dt*1000, blit=True)
plt.show()
""" progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
ani.save(filename="complexOscillator.mp4", fps=60, dpi=200, progress_callback=progress_callback, bitrate=5000) """