import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import cvxpy as cp

# ============================================
# 1. Path generation
# ============================================
def cubic_bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)
    curve = (
        ((1 - t) ** 3)[:, None] * p0 +
        (3 * ((1 - t) ** 2) * t)[:, None] * p1 +
        (3 * (1 - t) * (t ** 2))[:, None] * p2 +
        (t ** 3)[:, None] * p3
    )
    return curve[:, 0], curve[:, 1]


def generate_path():
    # Segment 1: straight
    x1 = np.linspace(0, 18, 120)
    y1 = np.zeros_like(x1)

    # Segment 2: descending curve
    p0 = np.array([18.0,  0.0])
    p1 = np.array([28.0,  0.5])
    p2 = np.array([36.0, -7.5])
    p3 = np.array([48.0, -11.0])
    x2, y2 = cubic_bezier(p0, p1, p2, p3, n=180)

    # Segment 3: long bottom curve
    p0 = np.array([48.0, -11.0])
    p1 = np.array([56.0, -12.5])
    p2 = np.array([62.0, -10.5])
    p3 = np.array([67.0, -4.0])
    x3, y3 = cubic_bezier(p0, p1, p2, p3, n=180)

    # Segment 4: aggressive upward turn
    p0 = np.array([67.0, -4.0])
    p1 = np.array([69.0,  1.0])
    p2 = np.array([69.5, 10.0])
    p3 = np.array([66.0, 16.0])
    x4, y4 = cubic_bezier(p0, p1, p2, p3, n=180)

    x = np.concatenate([x1, x2[1:], x3[1:], x4[1:]])
    y = np.concatenate([y1, y2[1:], y3[1:], y4[1:]])

    return x, y


def compute_path_geometry(path_x, path_y):
    dx = np.gradient(path_x)
    dy = np.gradient(path_y)
    psi = np.arctan2(dy, dx)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    kappa = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)

    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    s -= s[0]
    return psi, kappa, s


# ============================================
# 2. Utility functions
# ============================================
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def nearest_point_index(px, py, path_x, path_y):
    d2 = (path_x - px) ** 2 + (path_y - py) ** 2
    return int(np.argmin(d2))


def signed_lateral_error(px, py, path_x, path_y, path_psi, idx):
    xr = path_x[idx]
    yr = path_y[idx]
    psi_r = path_psi[idx]

    dx = px - xr
    dy = py - yr

    nx = -np.sin(psi_r)
    ny =  np.cos(psi_r)
    return dx * nx + dy * ny


def vehicle_polygon(x, y, yaw, length=4.4, width=2.0):
    corners = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2]
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    pts = corners @ R.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


# ============================================
# 3. Vehicle model
# ============================================
class Vehicle:
    def __init__(self, x, y, yaw, v, L=2.7):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L
        self.delta = 0.0

    def step(self, delta, dt):
        self.delta = delta
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * np.tan(delta) * dt
        self.yaw = wrap_angle(self.yaw)


# ============================================
# 4. Linear MPC
# ============================================
def solve_mpc(vehicle, path_x, path_y, path_psi, path_kappa, path_s,
              dt, Np, Nc, delta_prev,
              Q=np.diag([25.0, 12.0]),
              Qf=np.diag([120.0, 60.0]),
              R=1.5,
              Rd=80.0,
              delta_max_deg=18.0):

    L = vehicle.L
    v = vehicle.v
    delta_max = np.deg2rad(delta_max_deg)

    idx0 = nearest_point_index(vehicle.x, vehicle.y, path_x, path_y)

    ey0 = signed_lateral_error(vehicle.x, vehicle.y, path_x, path_y, path_psi, idx0)
    epsi0 = wrap_angle(vehicle.yaw - path_psi[idx0])

    ds_path = np.mean(np.diff(path_s))
    idx_step = max(1, int((v * dt) / max(ds_path, 1e-4)))

    ref_idx = []
    delta_ff = []

    for k in range(Np):
        idx = min(idx0 + k * idx_step, len(path_x) - 2)
        ref_idx.append(idx)
        delta_ff.append(np.arctan(L * path_kappa[idx]))

    ref_idx = np.array(ref_idx, dtype=int)
    delta_ff = np.array(delta_ff)

    # Linearized error dynamics
    # ey_{k+1}   = ey_k + v*dt*epsi_k
    # epsi_{k+1} = epsi_k + (v/L)*dt*(delta_k - delta_ff_k)
    A = np.array([
        [1.0, v * dt],
        [0.0, 1.0]
    ])
    B = np.array([
        [0.0],
        [(v / L) * dt]
    ])

    x = cp.Variable((2, Np + 1))
    u = cp.Variable(Nc)

    cost = 0
    cons = [x[:, 0] == np.array([ey0, epsi0])]

    for k in range(Np):
        uk = u[min(k, Nc - 1)]
        uff = delta_ff[k]

        cons += [x[:, k + 1] == A @ x[:, k] + B.flatten() * (uk - uff)] #상태는 시스템 모델을 따라야 한다는 제약조건
        cons += [cp.abs(uk) <= delta_max] #조향각의 절대값은 최대 조향각을 넘지 않아야 한다는 제약조건


        cost += cp.quad_form(x[:, k], Q)
        cost += R * cp.square(uk)

        if k == 0:
            cost += Rd * cp.square(uk - delta_prev)
        elif k < Nc:
            cost += Rd * cp.square(u[k] - u[k - 1])

    cost += cp.quad_form(x[:, Np], Qf)

    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
  
    if u.value is None:
        return 0.0, None, ey0

    delta_cmd = float(u.value[0])

    pred_x = []
    pred_y = []

    x_pred = x.value
    for k in range(1, Np + 1):
        idx = ref_idx[min(k - 1, len(ref_idx) - 1)]
        xr = path_x[idx]
        yr = path_y[idx]
        psi_r = path_psi[idx]
        ey = x_pred[0, k]

        px = xr - ey * np.sin(psi_r)
        py = yr + ey * np.cos(psi_r)

        pred_x.append(px)
        pred_y.append(py)

    pred = {
        "pred_x": np.array(pred_x),
        "pred_y": np.array(pred_y),
        "ey0": ey0
    }

    return delta_cmd, pred, ey0


# ============================================
# 5. Simulation settings
# ============================================
dt = 0.1
sim_steps = 600
vehicle_speed = 7.0

Np_A = 3
Np_B = 15
Nc = 2

Q  = np.diag([25.0, 12.0])
Qf = np.diag([120.0, 60.0])
R  = 1.5
Rd = 80.0
delta_max_deg = 18.0

path_x, path_y = generate_path()
path_psi, path_kappa, path_s = compute_path_geometry(path_x, path_y)

veh_A = Vehicle(x=0.0, y=1.8, yaw=np.deg2rad(5.0), v=vehicle_speed)
veh_B = Vehicle(x=0.0, y=1.8, yaw=np.deg2rad(5.0), v=vehicle_speed)

delta_prev_A = 0.0
delta_prev_B = 0.0

history_A_x, history_A_y = [veh_A.x], [veh_A.y]
history_B_x, history_B_y = [veh_B.x], [veh_B.y]

err_A_hist, err_B_hist = [], []
steer_A_hist, steer_B_hist = [], []
time_hist = []

frames = []

goal_x = path_x[-1]
goal_y = path_y[-1]
goal_tol = 2.0


# ============================================
# 6. Run simulation
# ============================================
for step in range(sim_steps):
    delta_A, pred_A, eyA = solve_mpc(
        veh_A, path_x, path_y, path_psi, path_kappa, path_s,
        dt, Np_A, Nc, delta_prev_A,
        Q=Q, Qf=Qf, R=R, Rd=Rd, delta_max_deg=delta_max_deg
    )

    delta_B, pred_B, eyB = solve_mpc(
        veh_B, path_x, path_y, path_psi, path_kappa, path_s,
        dt, Np_B, Nc, delta_prev_B,
        Q=Q, Qf=Qf, R=R, Rd=Rd, delta_max_deg=delta_max_deg
    )

    err_A_hist.append(eyA)
    err_B_hist.append(eyB)
    steer_A_hist.append(np.rad2deg(delta_A))
    steer_B_hist.append(np.rad2deg(delta_B))
    time_hist.append(step * dt)

    frames.append({
        "step": step,
        "time": step * dt,

        "vehA_x": veh_A.x,
        "vehA_y": veh_A.y,
        "vehA_yaw": veh_A.yaw,
        "histA_x": history_A_x.copy(),
        "histA_y": history_A_y.copy(),
        "predA_x": pred_A["pred_x"] if pred_A is not None else np.array([]),
        "predA_y": pred_A["pred_y"] if pred_A is not None else np.array([]),
        "eyA": eyA,
        "deltaA_deg": np.rad2deg(delta_A),
        "errA_hist": err_A_hist.copy(),
        "steerA_hist": steer_A_hist.copy(),

        "vehB_x": veh_B.x,
        "vehB_y": veh_B.y,
        "vehB_yaw": veh_B.yaw,
        "histB_x": history_B_x.copy(),
        "histB_y": history_B_y.copy(),
        "predB_x": pred_B["pred_x"] if pred_B is not None else np.array([]),
        "predB_y": pred_B["pred_y"] if pred_B is not None else np.array([]),
        "eyB": eyB,
        "deltaB_deg": np.rad2deg(delta_B),
        "errB_hist": err_B_hist.copy(),
        "steerB_hist": steer_B_hist.copy(),

        "time_hist": time_hist.copy()
    })

    veh_A.step(delta_A, dt)
    veh_B.step(delta_B, dt)

    delta_prev_A = delta_A
    delta_prev_B = delta_B

    history_A_x.append(veh_A.x)
    history_A_y.append(veh_A.y)
    history_B_x.append(veh_B.x)
    history_B_y.append(veh_B.y)

    dist_A = np.hypot(veh_A.x - goal_x, veh_A.y - goal_y)
    dist_B = np.hypot(veh_B.x - goal_x, veh_B.y - goal_y)

    if dist_A < goal_tol and dist_B < goal_tol:
        print(f"Simulation terminated early at step {step}: both vehicles reached goal region.")
        break


# ============================================
# 7. Visualization
# ============================================
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#111111")
gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.0, 1.0])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
    ax.set_facecolor("#111111")
    ax.grid(True, color='gray', alpha=0.3)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color("white")

for ax in (ax1, ax2):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(np.min(path_x) - 2, np.max(path_x) + 2)
    ax.set_ylim(np.min(path_y) - 3, np.max(path_y) + 3)
    ax.set_xlabel("X [m]", color='white')
    ax.set_ylabel("Y [m]", color='white')

ax1.set_title(f"Vehicle A: Short Horizon (Np={Np_A}, Nc={Nc})", color='white')
ax2.set_title(f"Vehicle B: Long Horizon (Np={Np_B}, Nc={Nc})", color='white')

t_max = max(1.0, len(frames) * dt)
all_err = np.array(err_A_hist + err_B_hist)
err_lim = max(2.5, np.max(np.abs(all_err)) + 0.2 if len(all_err) > 0 else 2.5)

for ax in (ax3, ax4):
    ax.set_xlim(0, t_max)
    ax.set_ylim(-err_lim, err_lim)
    ax.set_xlabel("Time [s]", color='white')
    ax.set_ylabel("Lateral Error [m]", color='white')
    ax.axhline(0, color='white', linestyle='--', linewidth=1.0, alpha=0.7)

for ax in (ax5, ax6):
    ax.set_xlim(0, t_max)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Time [s]", color='white')
    ax.set_ylabel("Steering [deg]", color='white')
    ax.axhline(0, color='white', linestyle='--', linewidth=1.0, alpha=0.7)

ax3.set_title("Lateral Error of Vehicle A", color='white')
ax4.set_title("Lateral Error of Vehicle B", color='white')
ax5.set_title("Steering of Vehicle A", color='white')
ax6.set_title("Steering of Vehicle B", color='white')

ax1.plot(path_x, path_y, '--', color='white', linewidth=2.0, label='Reference Path')
ax2.plot(path_x, path_y, '--', color='white', linewidth=2.0, label='Reference Path')

ax1.plot(goal_x, goal_y, 'r*', markersize=12, label='Goal')
ax2.plot(goal_x, goal_y, 'r*', markersize=12, label='Goal')

trajA_line, = ax1.plot([], [], color='deepskyblue', linewidth=2.5, label='Vehicle Trajectory')
predA_line, = ax1.plot([], [], 'o-', color='orange', linewidth=1.5, markersize=3, label='Predicted Horizon')
pointA, = ax1.plot([], [], 'o', color='cyan', markersize=7)
patchA = Polygon(np.zeros((4, 2)), closed=True, color='dodgerblue', alpha=0.9)
ax1.add_patch(patchA)

trajB_line, = ax2.plot([], [], color='lime', linewidth=2.5, label='Vehicle Trajectory')
predB_line, = ax2.plot([], [], 'o-', color='magenta', linewidth=1.5, markersize=3, label='Predicted Horizon')
pointB, = ax2.plot([], [], 'o', color='yellow', markersize=7)
patchB = Polygon(np.zeros((4, 2)), closed=True, color='mediumseagreen', alpha=0.9)
ax2.add_patch(patchB)

errA_line, = ax3.plot([], [], color='deepskyblue', linewidth=2.5)
errA_pt, = ax3.plot([], [], 'o', color='cyan', markersize=6)
errB_line, = ax4.plot([], [], color='lime', linewidth=2.5)
errB_pt, = ax4.plot([], [], 'o', color='yellow', markersize=6)

steerA_line, = ax5.plot([], [], color='deepskyblue', linewidth=2.2)
steerA_pt, = ax5.plot([], [], 'o', color='cyan', markersize=6)
steerB_line, = ax6.plot([], [], color='lime', linewidth=2.2)
steerB_pt, = ax6.plot([], [], 'o', color='yellow', markersize=6)

infoA = ax1.text(
    0.02, 0.98, "", transform=ax1.transAxes,
    va='top', ha='left', color='white',
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='white')
)
infoB = ax2.text(
    0.02, 0.98, "", transform=ax2.transAxes,
    va='top', ha='left', color='white',
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='white')
)

for ax in (ax1, ax2):
    leg = ax.legend(facecolor="#222222", edgecolor='white')
    for txt in leg.get_texts():
        txt.set_color("white")


def init():
    trajA_line.set_data([], [])
    predA_line.set_data([], [])
    pointA.set_data([], [])
    patchA.set_xy(vehicle_polygon(veh_A.x, veh_A.y, veh_A.yaw))
    errA_line.set_data([], [])
    errA_pt.set_data([], [])
    steerA_line.set_data([], [])
    steerA_pt.set_data([], [])

    trajB_line.set_data([], [])
    predB_line.set_data([], [])
    pointB.set_data([], [])
    patchB.set_xy(vehicle_polygon(veh_B.x, veh_B.y, veh_B.yaw))
    errB_line.set_data([], [])
    errB_pt.set_data([], [])
    steerB_line.set_data([], [])
    steerB_pt.set_data([], [])

    infoA.set_text("")
    infoB.set_text("")

    return (
        trajA_line, predA_line, pointA, patchA, infoA,
        trajB_line, predB_line, pointB, patchB, infoB,
        errA_line, errA_pt, errB_line, errB_pt,
        steerA_line, steerA_pt, steerB_line, steerB_pt
    )


def update(i):
    d = frames[i]

    trajA_line.set_data(d["histA_x"], d["histA_y"])
    predA_line.set_data(d["predA_x"], d["predA_y"])
    pointA.set_data([d["vehA_x"]], [d["vehA_y"]])
    patchA.set_xy(vehicle_polygon(d["vehA_x"], d["vehA_y"], d["vehA_yaw"]))

    trajB_line.set_data(d["histB_x"], d["histB_y"])
    predB_line.set_data(d["predB_x"], d["predB_y"])
    pointB.set_data([d["vehB_x"]], [d["vehB_y"]])
    patchB.set_xy(vehicle_polygon(d["vehB_x"], d["vehB_y"], d["vehB_yaw"]))

    errA_line.set_data(d["time_hist"], d["errA_hist"])
    errA_pt.set_data([d["time"]], [d["eyA"]])
    errB_line.set_data(d["time_hist"], d["errB_hist"])
    errB_pt.set_data([d["time"]], [d["eyB"]])

    steerA_line.set_data(d["time_hist"], d["steerA_hist"])
    steerA_pt.set_data([d["time"]], [d["deltaA_deg"]])
    steerB_line.set_data(d["time_hist"], d["steerB_hist"])
    steerB_pt.set_data([d["time"]], [d["deltaB_deg"]])

    infoA.set_text(
        f"step = {d['step']}\n"
        f"speed = {vehicle_speed:.2f} m/s\n"
        f"Np = {Np_A}, Nc = {Nc}\n"
        f"steering = {d['deltaA_deg']:.1f} deg\n"
        f"lateral error = {d['eyA']:.3f} m"
    )
    infoB.set_text(
        f"step = {d['step']}\n"
        f"speed = {vehicle_speed:.2f} m/s\n"
        f"Np = {Np_B}, Nc = {Nc}\n"
        f"steering = {d['deltaB_deg']:.1f} deg\n"
        f"lateral error = {d['eyB']:.3f} m"
    )

    return (
        trajA_line, predA_line, pointA, patchA, infoA,
        trajB_line, predB_line, pointB, patchB, infoB,
        errA_line, errA_pt, errB_line, errB_pt,
        steerA_line, steerA_pt, steerB_line, steerB_pt
    )


ani = FuncAnimation(
    fig, update, frames=len(frames), init_func=init,
    interval=120, blit=False, repeat=False
)

plt.tight_layout()
plt.show()