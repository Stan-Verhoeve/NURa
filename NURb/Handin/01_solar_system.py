import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy.constants import G
from helperscripts.integrate import rk4_single

# Constants
NAMES = np.array(["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"])
GRAV_CONST = G.to("AU3 / (solMass day2)").value

# Masses is Solar mass
MASSES = np.array([1.0, 1.651e-7, 2.447e-6, 3.003e-6, 3.213e-7, 9.545e-4, 2.857e-4, 4.365e-5, 5.150e-5])

def unique_pairs(N):
    # Number of indices to calculate
    Nidx = N * (N - 1) // 2
    i_idx = np.empty(Nidx, dtype=np.int32)
    j_idx = np.empty(Nidx, dtype=np.int32)

    # Compute all indices i < j
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            i_idx[k] = i
            j_idx[k] = j
            k += 1

    return i_idx, j_idx

def get_solar_initial(t):
    system = np.zeros((len(NAMES), 2, 3))

    # Initialize planets
    # TODO: Because vdesk doesn't like jpl
    # with solar_system_ephemeris.set("jpl"):
    with solar_system_ephemeris.set("builtin"):
        for i, name in enumerate(NAMES):
            pos, vel = get_body_barycentric_posvel(name, t)
            system[i,0,:] = pos.xyz.to_value(u.AU)
            system[i,1,:] = vel.xyz.to_value(u.AU / u.d)
    
    return system

def grav_acc(positions, masses):
    N = positions.shape[0]
    acc = np.zeros_like(positions)

    i, j = unique_pairs(N)

    dr = positions[i] - positions[j]
    dist2 = np.sum(dr**2, axis=1)
    dist3 = dist2 * np.sqrt(dist2)

    F = -GRAV_CONST * dr / dist3[:, np.newaxis]
    Fi = masses[j, np.newaxis] * F
    Fj = masses[i, np.newaxis] * F


    # Accumulate
    for n in range(len(i)):
        acc[i[n]] += Fi[n]
        acc[j[n]] -= Fj[n]

    return acc

def plot_system(time, positions, savename, names, **kwargs):
    fig, ax = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
    for i, obj in enumerate(names):
        ax[0].plot(positions[:,i,0], positions[:,i,1], label=obj, **kwargs)
        ax[1].plot(time, positions[:,i,2], label=obj, **kwargs)
    ax[0].set_aspect("equal", "box")
    ax[0].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set(xlabel="Time [yr]", ylabel="Z [AU]")
    plt.legend(loc=(1.05,0))
    plt.savefig(savename, bbox_inches="tight", dpi=600)
    plt.close()


def flatten_state(positions, velocities):
    return np.vstack([positions, velocities]).flatten()

def unflatten_state(state):
    N = len(state) // 6
    positions = state[:3*N].reshape(N, 3)
    velocities = state[3*N:].reshape(N, 3)
    return positions, velocities

def derivatives(t, state):
    positions, velocities = unflatten_state(state)
    acc = grav_acc(positions, MASSES)
    dpos_dt = velocities
    dvel_dt = acc
    return flatten_state(dpos_dt, dvel_dt)

def system(t, state):
    N = len(state) // 6
    pos, vel = unflatten_state(state)
    acc = grav_acc(pos, MASSES)
    dpos_dt = vel
    dvel_dt = acc
    return np.vstack([dpos_dt, dvel_dt]).flatten()

def get_orbital_energy(state, masses):
    pos, vel = unflatten_state(state)
    N = pos.shape[0]
    
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)

    potential = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dr = pos[i] - pos[j]
            r2 = np.dot(dr, dr)
            potential -= GRAV_CONST * masses[i] * masses[j] / np.sqrt(r2)

    return kinetic + potential

def make_movie(time, positions, savename, duration=30, fps=30, make_3d=False):
    # TODO: Currently hard-codes zoomed-in portion. Change later??

    import subprocess
    Npoints, Nbodies, Ndim = positions.shape
    Nframes = int(duration * fps)
    frame_indices = np.linspace(0, Npoints, Nframes, endpoint=False, dtype=int)
    
    mins = np.min(positions.reshape(-1, Ndim), axis=0)
    maxs = np.max(positions.reshape(-1, Ndim), axis=0)
    mins_zoom = np.min(positions[:,:5,:].reshape(-1, Ndim), axis=0)
    maxs_zoom = np.max(positions[:,:5,:].reshape(-1, Ndim), axis=0)

    if not make_3d:
        positions = positions[:,:,:-1]

    for fi, idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(121, projection="3d" if make_3d else None)
        ax2 = fig.add_subplot(122, projection="3d" if make_3d else None)
        
        ax1.set_xlim(mins[0], maxs[0])
        ax1.set_ylim(mins[1], maxs[1])
        ax2.set_xlim(mins_zoom[0], maxs_zoom[0])
        ax2.set_ylim(mins_zoom[1], maxs_zoom[1])

        ax1.set_xlabel("X [AU]")
        ax1.set_ylabel("Y [AU]")
        ax2.set_xlabel("X [AU]")
        ax2.set_ylabel("Y [AU]")
        
        if make_3d:
            ax1.set_zlim(mins[2], maxs[2])
            ax2.set_zlim(mins_zoom[2], maxs_zoom[2])
            
            ax1.set_zlabel("Z [AU]")
            ax2.set_zlabel("Z [AU]")
            
            angle = 360 * fi / Nframes
            ax1.view_init(elev=30, azim=angle)
            ax2.view_init(elev=30, azim=angle)
        
        for i in range(Nbodies):
            trail = positions[:idx+1, i]
            curr = positions[idx, i]
            
            ax1.plot(*trail.T, alpha=0.3)
            ax1.scatter(*curr.T, s=10)

            if i < 5:
                ax2.plot(*trail.T, alpha=0.3)
                ax2.scatter(*curr.T, s=10)
        
        ax1.set_title("All planets")
        ax2.set_title("4 innermost planets")
        fig.suptitle(f"t = {time[idx]:.2f} years")
        fig.tight_layout()
        # TODO: because vdesk is slow
        # plt.savefig(f"figures/movie/frame_{fi:04d}.png", dpi=300)
        plt.savefig(f"figures/movie/frame_{fi:04d}.png", dpi=100)
        plt.close(fig)
        
    
    if ".mp4" not in savename:
        savename += ".mp4"
    
    # TODO: Move this to ./run.sh??
    # Actually build the movie
    subprocess.run(["ffmpeg", "-framerate", f"{fps}", 
                              "-i", "figures/movie/frame_%04d.png",
                              "-pix_fmt", "yuv420p",
                              savename])


def main():
    # Time at which to grab initial conditions
    t = Time("2021-12-07 10:00")

    # Initial conditions
    initial_system = get_solar_initial(t)
    
    #############################
    ## Plotting initial system ##
    #############################
    fig, ax = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    for i, obj in enumerate(NAMES):
        pos, vel = initial_system[i]
        ax[0].scatter(pos[0], pos[1], label=obj)
        ax[1].scatter(pos[0], pos[2], label=obj)

    ax[0].set_aspect("equal", "box")
    ax[1].set_aspect("equal", "box")
    ax[0].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set(xlabel="X [AU]", ylabel="Z [AU]")
    plt.legend(loc=(1.05,0))
    plt.savefig("figures/Q1a", bbox_inches="tight", dpi=600)
    plt.close()
    
    ##########################
    ## Leapfrog integration ##
    ##########################
    # TODO: move to function??

    # Timings, units of day
    Tmax = 200 * u.yr.to("day")
    dt = 0.5
    Nsteps = int(Tmax / dt)
    time = np.linspace(dt * u.day.to("year"), Tmax * u.day.to("year"), Nsteps)
    Nobj = len(NAMES)
        
    # Positions, velocities, and energies
    lf_positions = np.zeros((Nsteps, Nobj, 3))
    lf_velocities = np.zeros((Nsteps, Nobj, 3))
    lf_energy = np.zeros(Nsteps)

    rk_states = np.zeros((Nsteps, Nobj*6))
    rk_energy = np.zeros(Nsteps)
    
    # Initial state
    lf_positions[0] = initial_system[:,0,:]
    lf_velocities[0] = initial_system[:,1,:]
    rk_states[0] = flatten_state(initial_system[:,0,:], initial_system[:,1,:])
    # TODO: because vdesk is slow
    # rk_energy[0] = get_orbital_energy(rk_states[0], MASSES)
    # lf_energy[0] = get_orbital_energy(rk_states[0], MASSES)
    
    # First kick with half timestep
    acc = grav_acc(lf_positions[0], MASSES)
    lf_velocities[0] += 0.5 * dt * acc

    for i in range(Nsteps-1):
        ##############
        ## Leapfrog ##
        ##############

        # Update positions
        lf_positions[i+1] = lf_positions[i] + dt * lf_velocities[i]
        
        # Update velocities (with new accelerations)
        acc_new = grav_acc(lf_positions[i+1], MASSES)
        lf_velocities[i+1] = lf_velocities[i] + dt * acc_new

        # TODO: because vdesk is slow
        # lf_state = flatten_state(lf_positions[i+1], lf_velocities[i+1])
        # lf_energy[i+1] = get_orbital_energy(lf_state, MASSES)

        #########
        ## RK4 ##
        #########
        rk_states[i+1] = rk4_single(system, time[i], rk_states[i], dt)
        
        # TODO: because vdesk is slow
        # rk_energy[i+1] = get_orbital_energy(rk_states[i+1], MASSES)
        
    # Convert 1D state to positions and velocities
    rk_positions = rk_states[:, :3*Nobj].reshape(Nsteps, -1, 3)
    rk_velocities = rk_states[:, 3*Nobj:].reshape(Nsteps, -1, 3)
    
    # Plot systems
    plot_system(time, lf_positions, "figures/Q1b", NAMES)
    plot_system(time, lf_positions, "figures/Q1b_zoomed", NAMES[:5], alpha=0.3)
    plot_system(time, rk_positions, "figures/Q1c", NAMES)
    plot_system(time, rk_positions, "figures/Q1c_zoomed", NAMES[:5], alpha=0.3)

    # Comparison plot
    differences = np.abs(rk_positions[:,:,0] - lf_positions[:,:,0])
    fig, ax = plt.subplots(1,3, figsize=(12,5), constrained_layout=True)
    for i, obj in enumerate(NAMES):
        abs_diff = differences[:,i]
        ax[0].plot(time, lf_positions[:,i,0], label=obj, alpha=0.3)
        ax[1].plot(time, rk_positions[:,i,0], label=obj, alpha=0.3)
        ax[2].plot(time, abs_diff, label=obj, alpha=0.3)
    ax[0].set(xlabel="Time [yr]", ylabel="X [AU]", title="Leapfrog")
    ax[1].set(xlabel="Time [yr]", ylabel="X [AU]", title="RK4")
    ax[2].set(xlabel="Time [yr]", ylabel=r"|$x_{RK} - x_{LF}$|", title="Diffence between methods")
    plt.legend(loc=(1.05,0))
    plt.savefig("figures/Q1c_comparison", bbox_inches="tight", dpi=600)
    plt.close()
    
    # TODO: removed because vdesk is slow
    # Energy plot
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(time, lf_energy, label="Leapfrog", alpha=0.3)
    # ax.plot(time, rk_energy, label="RK", alpha=0.3)
    # ax.set(title="Total energy in sytem",
    #        xlabel="Time [yr]",
    #        ylabel="Energy [J]",
    #        )
    # ax.legend()
    # fig.savefig("figures/Q1c_energy", bbox_inches="tight", dpi=600)
    # plt.close()
    
    ###########
    ## Bonus ##
    ###########
    # make_movie(time, lf_positions[:,:5,:], "figures/Q1_movie_2d.mp4", duration=10, make_3d=False)
    make_movie(time, lf_positions, "figures/Q1_movie_3d.mp4", duration=10, make_3d=True)
if __name__ in ("__main__"):
    main()
