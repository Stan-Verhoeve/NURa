import numpy as np
import h5py
import matplotlib.pyplot as plt

# Question 2: Calculating potentials

with h5py.File("/disks/cosmodm/DMO_a0.1_256.hdf5","r") as handle:
    pos=handle["Position"][...] #particle positions, shape (Np,3), comoving
    #vel=handle["Velocity"][...] #particle velocities, shape (Np,3), comoving <-- not used, but if you're interested

Np=np.int64(256)**3 #number of particles
mp=np.float32(3.64453e10) #particle mass in Msun; all 32-bit to save memory
G=np.float32(4.3009e-9) #gravitational constant in Mpc*(km/s)^2/Msun
h=np.float32(0.3755) #Hubble parameter (this is a Einstein-de Sitter universe with Omega_m=1)
L=np.float32(250.0) #side length of periodic cubic simulation volume
scale_factor=np.float32(0.1) #scale factor a
redshift=1.0/scale_factor-1
rho_mean=Np*mp/L**3 #mean density in Msun/Mpc^3 (comoving, matches 3*H_0^2/(8*pi*G))

# Question 2a: using Barnes-Hut [note: not actually calculating a potential, unless you do the bonus question]

# TO DO: build an octree (use a class for a node, so it can also refer to child nodes; avoid using lists for anything or the memory will balloon)

# Plotting the mass distribution for a slice

for level in [3,5,7]: #feel free to change any of this code
    pixels=2**level
    massmap=np.zeros(4,(pixels,pixels),dtype=np.float32)
    # TO DO: traverse the octree, fill map massmap[0,:,:] with the masses of nodes at depth 3 and x_index=x_0,
    #        massmap[1,:,:] with the masses of nodes at depth 3 and x_index=x_1, etc; then plot these slices;
    #        then do the same for levels 5 and 7

    fig, ax = plt.subplots(2,2, figsize=(10,8))
    pcm = ax[0,0].pcolormesh(np.arange(pixels), np.arange(pixels), massmap[0,:,:])
    #ax[0,0].set(ylabel='...', title='...')
    fig.colorbar(pcm, ax=ax[0,0], label='Total mass inside node')
    pcm =ax[0,1].pcolormesh(np.arange(pixels), np.arange(pixels), massmap[1,:,:])
    #ax[0,1].set(title='...')
    fig.colorbar(pcm, ax=ax[0,1], label='Total mass inside node')
    pcm = ax[1,0].pcolormesh(np.arange(pixels), np.arange(pixels), massmap[2,:,:])
    #ax[1,0].set(ylabel='...', xlabel='...', title='...')
    fig.colorbar(pcm, ax=ax[1,0], label='Total mass inside node')
    pcm = ax[1,1].pcolormesh(np.arange(pixels), np.arange(pixels), massmap[3,:,:])
    #ax[1,1].set(xlabel='...', title='...')
    fig.colorbar(pcm, ax=ax[1,1], label='Total mass inside node')
    ax[0,0].set_aspect('equal', 'box')
    ax[0,1].set_aspect('equal', 'box')
    ax[1,0].set_aspect('equal', 'box')
    ax[1,1].set_aspect('equal', 'box')
    plt.savefig(f"fig2b_level{level}.png",dpi=300)
    plt.close()

# Question 2b: using the FFT

Ngrid=np.int64(128)
densgrid=np.zeros((Ngrid,Ngrid,Ngrid),dtype=np.float32)
potential=np.zeros((Ngrid,Ngrid,Ngrid),dtype=np.float32)
# TO DO: assign particle masses to densgrid, convert to density, and calculate potentials from it

# Plotting four slices of a grid

fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[0,:,:])
#ax[0,0].set(ylabel='...', title='...')
fig.colorbar(pcm, ax=ax[0,0], label='Potential')
pcm =ax[0,1].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[16,:,:])
#ax[0,1].set(title='...')
fig.colorbar(pcm, ax=ax[0,1], label='Potential')
pcm = ax[1,0].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[32,:,:])
#ax[1,0].set(ylabel='...', xlabel='...', title='...')
fig.colorbar(pcm, ax=ax[1,0], label='Potential')
pcm = ax[1,1].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[64,:,:])
#ax[1,1].set(xlabel='...', title='...')
fig.colorbar(pcm, ax=ax[1,1], label='Potential')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2b.png",dpi=300)
plt.close()
