import numpy as np
import matplotlib.pyplot as plt

###############
## Constants ##
###############
MSTAR = 1.
MJ = 9.55e-4 * MSTAR
G = 1.

###############
## functions ##
###############
def orbital_radius(period, G, M):
    """Orbital radius asusming circular orbits"""
    return (G * M * period ** 2 / (4 * np.pi ** 2)) ** (1/3)

def grav_acc(r):
    dist = np.sqrt(np.sum(r**2))
    return -G * MSTAR * r / dist ** 3

def system(t, state):
    r = state[:2]
    v = state[2:]
    acc = grav_acc(r)

    return np.hstack((v, acc))

def get_energy(state, m):
    r = state[:2]
    v = state[2:]
    Ek = 0.5 * m * np.sum(v**2)
    Ep = -G * MSTAR * m / np.sqrt(np.sum(r**2))

    return Ek + Ep

def main():
    from helperscripts.integrate import solve_ivp, euler_single, rk4_single
    USE_RK = False

    # Planet properties
    M1 = MJ
    M2 = 0.011 * MJ
    P1 = 12.
    P2 = 1.

    # Planet initial conditions
    r1 = orbital_radius(P1, G, MSTAR)
    r2 = orbital_radius(P2, G, MSTAR)
    v1 = np.sqrt(G * MSTAR / r1)
    v2 = np.sqrt(G * MSTAR / r2)
    
    # Timings
    dt = 1e-2
    tmax = 120.
    Nsteps = int(tmax / dt)

    # Initialize arrays
    pos1_leapfrog = np.zeros((Nsteps, 2))
    vel1_leapfrog = np.zeros((Nsteps, 2))
    pos2_leapfrog = np.zeros((Nsteps, 2))
    vel2_leapfrog = np.zeros((Nsteps, 2))
    
    # Keep track of orbital energies
    Etot1_euler = np.zeros(Nsteps)
    Etot2_euler = np.zeros(Nsteps)
    Etot1_leapfrog = np.zeros(Nsteps)
    Etot2_leapfrog = np.zeros(Nsteps)
    
    # States for solving using Euler
    state1 = np.zeros((Nsteps, 4))
    state2 = np.zeros((Nsteps, 4))

    # Initial conditions
    pos1_leapfrog[0] = [r1, 0.0]
    vel1_leapfrog[0] = [0.0, v1]
    pos2_leapfrog[0] = [r2, 0.0]
    vel2_leapfrog[0] = [0.0, v2]

    state1[0] = [r1, 0.0, 0.0, v1]
    state2[0] = [r2, 0.0, 0.0, v2]

    Etot1_euler[0] = get_energy(state1[0], M1)
    Etot2_euler[0] = get_energy(state2[0], M2)
    Etot1_leapfrog[0] = get_energy(state1[0], M1)
    Etot2_leapfrog[0] = get_energy(state2[0], M2)
    
    # Integrate
    for i in range(Nsteps-1):
        t = i * dt
        # Euler integration
        if USE_RK:
            state1[i+1] = rk4_single(system, t, state1[i], dt)
            state2[i+1] = rk4_single(system, t, state2[i], dt)
        else:
            state1[i+1] = euler_single(system, t, state1[i], dt)
            state2[i+1] = euler_single(system, t, state2[i], dt)
        
        # Track orbital energy
        Etot1_euler[i+1] = get_energy(state1[i+1], M1)
        Etot2_euler[i+1] = get_energy(state2[i+1], M2)

        # Leapfrog integration
        a1 = grav_acc(pos1_leapfrog[i])
        a2 = grav_acc(pos2_leapfrog[i])
        
        v1_half = vel1_leapfrog[i] + 0.5 * dt * a1
        pos1_leapfrog[i+1] = pos1_leapfrog[i] + dt * v1_half
        a1_new = grav_acc(pos1_leapfrog[i+1])
        vel1_leapfrog[i+1] = v1_half + 0.5 * dt * a1_new
        
        # Track orbital energy (planet 1)
        state = np.concatenate([pos1_leapfrog[i+1], vel1_leapfrog[i+1]])
        Etot1_leapfrog[i+1] = get_energy(state, M1)
        
        v2_half = vel2_leapfrog[i] + 0.5 * dt * a2
        pos2_leapfrog[i+1] = pos2_leapfrog[i] + dt * v2_half
        a2_new = grav_acc(pos2_leapfrog[i+1])
        vel2_leapfrog[i+1] = v2_half + 0.5 * dt * a2_new
        
        # Track orbital energy (planet 2)
        state = np.concatenate([pos2_leapfrog[i+1], vel2_leapfrog[i+1]])
        Etot2_leapfrog[i+1] = get_energy(state, M2)

    # Euler integration using solve_ivp
    y0_p1 = [r1, 0, 0, v1]
    y0_p2 = [r2, 0, 0, v2]
    __, sols1 = solve_ivp(system, (0,tmax), y0_p1, dt, method="euler")
    __, sols2 = solve_ivp(system, (0,tmax), y0_p2, dt, method="euler")
    
    ##############
    ## Plotting ##
    ##############
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221, aspect=1)
    ax2 = fig.add_subplot(222, aspect=1, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    if USE_RK:
        label = "Forward RK4"
    else:
        label = "Forward Euler"
    ax1.plot(*state1[:,:2].T, c="r", label=label)
    # ax1.plot(*sols1[:,:2].T, c="r", label="Forward Euler")
    ax1.plot(*pos1_leapfrog.T, c="k", ls="--", label="Leapfrog")
    
    ax2.plot(*state2[:,:2].T, c="r")
    # ax2.plot(*sols2[:,:2].T, c="r")
    ax2.plot(*pos2_leapfrog.T, c="k", ls="--")
    
    ax3.plot(Etot1_euler, c="r")
    ax3.plot(Etot1_leapfrog, c="k", ls="--")
    ax4.plot(Etot2_euler, c="r")
    ax4.plot(Etot2_leapfrog, c="k", ls="--")

    ax1.set(title=f"Planet 1, P = {P1} yr, M = {M1 / MJ} $M_J$",
            xlabel="x",
            ylabel="y",
           )
    ax2.set(title=f"Planet 2, P = {P2} yr, M = {M2 / MJ} $M_J$",
            xlabel="x",
           )
    ax2.tick_params(labelleft=False)
    ax3.set(title=f"Planet 1, total orbital energy",
            xlabel="Step",
            ylabel="Etot [a.u.]",
            # yscale="log",
            )
    ax4.set(title=f"Planet 2, total orbital energy",
            xlabel="Step",
            ylabel="Etot [a.u.]",
            # yscale="log",
            )
    ax1.legend()
    
    fig.suptitle(f"Planetary orbits integrated for {tmax} years, dt = {dt} yr")
    
    fig.tight_layout()
    fig.savefig("figures/02_two_planets_around_star_Q2a", bbox_inches="tight", dpi=600)

if __name__ in ("__main__"):
    main()
