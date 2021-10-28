import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg

sys.path.append('../qc_dem/')

from qc_dem.boundary import Boundary
from qc_dem.particles import (
    Particles,
    NList,
)
from qc_dem.variational_int import (
    time_step,
    HookianContact,
    BondContact,
    SumContact,
    Wall,
    WallNList,
)

from utils import *

def setup_particles_from_liggghts(data, k=1e6, gamma=0):
    m = data['mass'][1]
    N = len(data['x'])
    print('N =', N)

    particles = Particles()
    particles.k = k
    particles.d = data['diameter'][1]
    particles.gamma_n = gamma * m/2
    particles.gamma_t = gamma * m/4

    particles.xyz = np.zeros((N, 3))
    particles.rot = np.zeros((N, 3))
    particles.p = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))

    particles.xyz[:, 0] = data['x'].to_numpy()
    particles.xyz[:, 1] = data['y'].to_numpy()
    particles.xyz[:, 2] = data['z'].to_numpy()

    particles.m = m * np.ones((N, 3))

    particles.p[:, 0] = m * data['vx'].to_numpy()
    particles.p[:, 1] = m * data['vy'].to_numpy()
    particles.p[:, 2] = m * data['vz'].to_numpy()

    particles.angular_mom[:, 0] = particles.mom_inertia[:, 0] * data['omegax'].to_numpy()
    particles.angular_mom[:, 1] = particles.mom_inertia[:, 1] * data['omegay'].to_numpy()
    particles.angular_mom[:, 2] = particles.mom_inertia[:, 2] * data['omegaz'].to_numpy()

    particles.updateM()
    return particles

def dump_file_timestep(filename):
    ## Gets the integer value of the timestep from a dump filename.
    s = filename.strip('block_bpm_').strip('.dump')
    return int(s)


def load(filename):
    data = load_liggghts_dump(filename)
    key = list(data.keys())[0]
    data = data[key]
    data['id'] = pd.to_numeric(data['id'], downcast='integer')
    data.sort_values('id', inplace=True)
    data.set_index('id', inplace=True)
    return data

def prepare_dataframe(N_particles, N_timesteps):
    df = pd.DataFrame()
    for i in range(N_particles):
        df['x' + str(i)] = np.zeros(N_timesteps)
        df['y' + str(i)] = np.zeros(N_timesteps)
        df['z' + str(i)] = np.zeros(N_timesteps)
        df['vx' + str(i)] = np.zeros(N_timesteps)
        df['vy' + str(i)] = np.zeros(N_timesteps)
        df['vz' + str(i)] = np.zeros(N_timesteps)
        df['omegax' + str(i)] = np.zeros(N_timesteps)
        df['omegay' + str(i)] = np.zeros(N_timesteps)
        df['omegaz' + str(i)] = np.zeros(N_timesteps)
    return df

def plot_trail(fig, x, z, trail_legend, particle_legend, label=None):

    tt = np.linspace(0, 2*np.pi, 101)
    xx = 0.5 * np.cos(tt)
    zz = 0.5 * np.sin(tt)

    #print(' ', particle_legend, x, z)
    fig.plot(x, z, trail_legend) #'C{0}-'.format(i))

    fig.plot(
        xx + x[-1],
        zz + z[-1],
        particle_legend,
    )
    if not label is None:
        plt.text(x[-1], z[-1], label)



def make_cache_filename(**kwargs):
    return '{name}_{T}_{dt}_{k}_{gamma}.csv'.format(**kwargs)

def get_lammps_dump_files(dt, dump_folder):

    ## Load lammps data
    dump_files = os.listdir(dump_folder)
    dump_files.sort(key=dump_file_timestep)
    #if 'box0.dump' == dump_files[0]:
    #    # Remove the 0 timestep file
    #    dump_files.pop(0)

    return dump_files

def calc_total_kinetic_E(m, d, N_particles, data):

    I = 2./5 * m * (d/2)**2

    data['K_lin'] = np.zeros(data.shape[0])
    data['K_rot'] = np.zeros(data.shape[0])
    for j in range(N_particles):
        data['K_lin'] += 0.5* m * (
            data['vx' + str(j)]**2 +
            data['vy' + str(j)]**2 +
            data['vz' + str(j)]**2)
        data['K_rot'] += 0.5 * I * (
            data['omegax'+str(j)]**2 +
            data['omegay'+str(j)]**2 +
            data['omegaz'+str(j)]**2
        )

def plot_lammps(dt, T, particles_to_plot, dump_folder, plot_xy, plot_zt, plot_K, plot_gran_temp, k, gamma):

    dump_files = get_lammps_dump_files(dt, dump_folder)
    # Starting configuration
    #data = load_liggghts_dump('liggghts/box/dump/' + dump_files[0])
    #data = data[1000]
    #data.sort_values('id', inplace=True)

    T0 = dt * dump_file_timestep(dump_files[0])

    dump_files = [
        dump_file
        for dump_file in dump_files
        if dump_file_timestep(dump_file) <= (T + T0)/dt + 0.5
    ]
    N_particles = 0
    lammps_data = None

    lammps_cache_filename = make_cache_filename(name='cache/box_lammps', T=T, dt=dt, k=k, gamma=gamma)
    if os.path.exists(lammps_cache_filename) and not '--no-lammps-cache' in sys.argv:
        print('Found LAMMPS cache: {0}'.format(lammps_cache_filename))
        lammps_data = pd.read_csv(lammps_cache_filename)

        data = load(dump_folder + dump_files[0])
        m = data['mass'].to_numpy()[0]
        d = data['diameter'].to_numpy()[0]
        N_particles = data.shape[0]
    else:
        print(dump_files)
        for i, dump_file in enumerate(dump_files):
            print('Loading: ', dump_folder+dump_file)
            data = load(dump_folder + dump_file)

            if lammps_data is None:
                N_particles = data.shape[0]
                lammps_data = prepare_dataframe(N_particles, len(dump_files))


            for j in range(N_particles):
                lammps_data['x' + str(j)][i] = data['x'][j+1]
                lammps_data['y' + str(j)][i] = data['y'][j+1]
                lammps_data['z' + str(j)][i] = data['z'][j+1]
                lammps_data['vx' + str(j)][i] = data['vx'][j+1]
                lammps_data['vy' + str(j)][i] = data['vy'][j+1]
                lammps_data['vz' + str(j)][i] = data['vz'][j+1]
                lammps_data['omegax' + str(j)][i] = data['omegax'][j+1]
                lammps_data['omegay' + str(j)][i] = data['omegay'][j+1]
                lammps_data['omegaz' + str(j)][i] = data['omegaz'][j+1]
        lammps_data['t'] = np.linspace(0, T, len(dump_files))
        lammps_data.to_csv(lammps_cache_filename)

        m = data['mass'].to_numpy()[0]
        d = data['diameter'].to_numpy()[0]

    lammps_data['ave_vx'] = np.zeros(lammps_data['t'].size)
    lammps_data['ave_vy'] = np.zeros(lammps_data['t'].size)
    lammps_data['ave_vz'] = np.zeros(lammps_data['t'].size)
    for i in range(N_particles):
        lammps_data['ave_vx'] += lammps_data['vx' + str(i)]/N_particles
        lammps_data['ave_vy'] += lammps_data['vy' + str(i)]/N_particles
        lammps_data['ave_vz'] += lammps_data['vz' + str(i)]/N_particles

    lammps_data['T_xx'] = np.zeros(lammps_data['t'].size)
    lammps_data['T_yy'] = np.zeros(lammps_data['t'].size)
    lammps_data['T_zz'] = np.zeros(lammps_data['t'].size)
    for i in range(N_particles):
        lammps_data['T_xx'] += (lammps_data['ave_vx'] - lammps_data['vx' + str(i)]/N_particles)**2/N_particles
        lammps_data['T_yy'] += (lammps_data['ave_vy'] - lammps_data['vy' + str(i)]/N_particles)**2/N_particles
        lammps_data['T_zz'] += (lammps_data['ave_vz'] - lammps_data['vz' + str(i)]/N_particles)**2/N_particles

    lammps_data['gran_temp'] = (1/3) * (
        lammps_data['T_xx'] +
        lammps_data['T_yy'] +
        lammps_data['T_zz']
    )
    plot_gran_temp.plot(
        lammps_data['t'],
        lammps_data['T_zz'],
        'C{0}--'.format(plot_gran_temp.n_plot),
        label='LAMMPS ($\Delta t = {0}$)'.format(dt)
    )


    calc_total_kinetic_E(m, d, N_particles, lammps_data)
    plot_K.plot(
        lammps_data['t'],
        lammps_data['K_lin'] + lammps_data['K_rot'],
        'C{0}--'.format(plot_K.n_plot),
        label='LAMMPS ($\Delta t = {0}$)'.format(dt)
    )

    T_scale = np.pi/np.sqrt(2*k/m)
    for i, part_i in enumerate(particles_to_plot):
        plot_trail(
            plot_xy,
            lammps_data['x'+str(part_i)].to_numpy(),
            lammps_data['z'+str(part_i)].to_numpy(),
            'k--',
            'k-',
            label=None,#str(part_i+1))
        )

        plot_zt.plot(
            lammps_data['t'].to_numpy(),
            lammps_data['z'+str(part_i)].to_numpy(),
            'C{0}--'.format(i))
        plt.text(
            lammps_data['t'].to_numpy()[-1],
            lammps_data['z'+str(part_i)].to_numpy()[-1],
            #'$\Delta t / T_c = {0:.3f}$'.format(dt/T_scale),
            '$T_c/\Delta t  = {0:.1f}$'.format(T_scale/dt),
        )




def main():

    #d = 0.25
    d = 1.
    L = 2
    L_zhi = 50
    k = 1000000
    dx = 0.05 * d
    rho = 1.000#*(2*np.sqrt(2))
    volume = 4./3. * np.pi * (d/2)**3
    m =  volume * rho

    gamma = 100
    #gamma_n = gamma * m
    #gamma_t = gamma_n/2

    dt = 0.001
    if '--profile' in sys.argv:
        T = 1000 * dt #0.5
    else:
        T = 2.0
        #T = 1000 * dt
    dt_dump = 1*dt #0.001
    dump_step = int(dt_dump/dt+0.5)
    #dt_dump = dump_step*dt

    bounds = Boundary(L, L, d)


    # Figures
    particles_to_plot = [0, 1, 14, 20, 777]
    fig_pos_xy = Figure('x', 'Position (y)', show=True)
    fig_K = Figure('t', 'K')
    fig_xt = Figure('t', 'x', show=False)
    fig_zt = Figure('t', 'z')
    fig_gran_temp = Figure('t', 'T')


    # Plot LAMMPS data
    '''
    T_lammps = 5
    if not '--profile' in sys.argv:
        #for lammps_dt in [0.0005, 0.0001, 0.00001, 0.000001, 0.0000001]:
        for lammps_dt in [0.0001, 0.00001, 0.000001]:
            dump_folder = 'liggghts/box/dump_{dt:.10f}'.format(dt=lammps_dt).strip('0')+'/'
            plot_lammps(lammps_dt, T_lammps, particles_to_plot, dump_folder, fig_pos_xy, fig_zt, fig_K, fig_gran_temp, k, gamma)
    '''

    dump_folder = 'liggghts/bpm_block/dump/'
    cache_filename = make_cache_filename(name='cache/block_bpm_particles', T=T, dt=dt, k=k, gamma=gamma)
    # Starting configuration
    dump_files = get_lammps_dump_files(dt, dump_folder)
    data = load(dump_folder + dump_files[0])

    # Create particles
    particles = setup_particles_from_liggghts(data, k=k, gamma=gamma)
    nlist = NList(d/2)

    # Setup the contact model
    bond_contact = BondContact(particles, dt)
    hookian_contact = HookianContact(particles, dt, bonds=bond_contact)
    contact = SumContact(particles, hookian_contact, bond_contact)

    N_particles = particles.xyz.shape[0]
    mass = particles.m[0,0]

    # Number of time steps used in plotting the figure (based on the largest time step)
    N_t_steps = int(T/dt)+1
    N_dump_steps = int(T/dt_dump)+1
    print('N_dump_steps:', N_dump_steps)


    if os.path.exists(cache_filename) and not '--no-cache' in sys.argv:
        print('Found box simulation cache: {0}'.format(cache_filename))
        figure_data = pd.read_csv(cache_filename)
    else:
        # Perform our DEM simulation

        figure_data = prepare_dataframe(N_particles, N_dump_steps)
        figure_data['t'] = np.linspace(0, T, N_dump_steps)

        ## Wall setup

        walls = [
            #Wall(np.array([-L, 0, 0]), np.array([ 1, 0, 0]), k, gamma * mass, gamma * mass/2),
            #Wall(np.array([ L, 0, 0]), np.array([-1, 0, 0]), k, gamma * mass, gamma * mass/2),
            Wall(np.array([0, -d/2, 0]), np.array([0,  1, 0]), k, 0, 0),
            Wall(np.array([0,  d/2, 0]), np.array([0, -1, 0]), k, 0, 0),
            Wall(np.array([0, 0, 0]), np.array([0, 0,  1]), k, gamma * mass, gamma * mass/2),
            #Wall(np.array([0, 0,  L_zhi]), np.array([0, 0, -1]), k, gamma * mass, gamma * mass/2),
           ]

        wall_nlist = WallNList(d/2, walls)

        t = 0
        step = 0
        i_dump = 0
        dr = np.zeros(particles.xyz.shape)
        while step <= N_t_steps:
            # Run the simulation for one time step
            nlist.time_step(particles, dr)
            wall_nlist.time_step(particles, dr)
            dr = time_step(
                particles,
                dt,
                walls=walls,
                gravity=np.array([0, 0, -9.8]),
                nlist=nlist,
                contact_law=contact,
            )

            if step % dump_step == 0:
                print(' t = {1:.3f} ({0:.2f}%)'.format(100.0*step/N_t_steps, step*dt))
                for i in range(N_particles):
                    figure_data['x' + str(i)][i_dump] = particles.xyz[i, 0]
                    figure_data['y' + str(i)][i_dump] = particles.xyz[i, 1]
                    figure_data['z' + str(i)][i_dump] = particles.xyz[i, 2]
                    figure_data['vx' + str(i)][i_dump] = particles.v[i, 0]
                    figure_data['vy' + str(i)][i_dump] = particles.v[i, 1]
                    figure_data['vz' + str(i)][i_dump] = particles.v[i, 2]
                    figure_data['omegax' + str(i)][i_dump] = particles.angular_v[i, 0]
                    figure_data['omegay' + str(i)][i_dump] = particles.angular_v[i, 1]
                    figure_data['omegaz' + str(i)][i_dump] = particles.angular_v[i, 2]
                i_dump += 1
            #elif step % 500:
            #    print(' t = {1:.3f} ({0:.2f}%)'.format(100.0*step/N_t_steps, step*dt))

            step += 1

        figure_data.to_csv(cache_filename)

    calc_total_kinetic_E(m, d, N_particles, figure_data)
    fig_K.plot(
        figure_data['t'],
        figure_data['K_lin'] + figure_data['K_rot'],
        'C{0}-'.format(fig_K.n_plot),
        label='Var. Int. ($\Delta t = {0}$)'.format(dt)
    )

    p_mag = np.sqrt(
        particles.p[:, 0]**2 +
        particles.p[:, 1]**2 +
        particles.p[:, 2]**2
    )
    #particles_to_plot = np.argsort(p_mag)
    #particles_to_plot = particles_to_plot[-5:]
    particles_to_plot = np.arange(particles.N)

    #for i in range(20):
    for i, part_i in enumerate(particles_to_plot):
        plot_trail(
            fig_pos_xy,
            figure_data['x'+str(part_i)].to_numpy(),
            figure_data['z'+str(part_i)].to_numpy(),
            'b--',
            'b-',
            label=None, #str(part_i+1)
        )
        for j in bond_contact.bonds[i]:
            if i < j:
                fig_pos_xy.plot(
                    [figure_data['x'+str(i)].to_numpy()[-1], figure_data['x'+str(j)].to_numpy()[-1]],
                    [figure_data['z'+str(i)].to_numpy()[-1], figure_data['z'+str(j)].to_numpy()[-1]],
                    'g--',
                )

        #plt.plot([L, L, -L, -L, L], [L_zhi, -L, -L, L_zhi, L_zhi], 'k:')
        #plt.plot([-3*L, 3*L], [0, 0], 'k:')
        #plt.axis('equal')
        continue

        fig_zt.plot(figure_data['t'].to_numpy(),
                    figure_data['z'+str(part_i)].to_numpy(),
                    'C{0}-'.format(i))

        plt.text(figure_data['t'].to_numpy()[-1],
                 figure_data['z'+str(part_i)].to_numpy()[-1],
                 str(part_i+1))



    save_tikz_cartoon(
        'figures/block_bpm_{0:f}.tex'.format(dt),
        particles,
        #drawvel=True,
        bonds=bond_contact.bonds,
        p_scale=np.max(particles.p[:]),
        y_coord_idx=2,
    )
    plt.figure(fig_pos_xy.fig.number)
    plt.plot([-3*L, 3*L], [0, 0], 'k:')
    plt.axis('equal')
    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
