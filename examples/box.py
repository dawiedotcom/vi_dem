import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import (
    ic,
    install
)
install()
ic.configureOutput(outputFunction=print)

sys.path.append('.')

from vi_dem.particles import (
    Particles,
    NList,
)
from vi_dem.variational_int import (
    time_step,
    HookianContact,
)
from vi_dem.wall import (
    Wall,
    WallNList,
)

from utils import *

def setup_particles_from_lammps(data, k=1e6, gamma=0):
    m = data['mass']#[1]
    N = len(data['x'])
    print('N =', N)

    particles = Particles()
    particles.k = k
    particles.d = data['diameter'].to_numpy() #[1]
    particles.gamma_n = gamma #* m/2
    particles.gamma_t = gamma / 2 # * m/4

    particles.xyz = np.zeros((N, 3))
    particles.rot = np.zeros((N, 3))
    particles.p = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))

    particles.xyz[:, 0] = data['x'].to_numpy()
    particles.xyz[:, 1] = data['y'].to_numpy()
    particles.xyz[:, 2] = data['z'].to_numpy()

    particles.m = m.to_numpy() #* np.ones((N, 3))

    particles.p[:, 0] = m * data['vx'].to_numpy()
    particles.p[:, 1] = m * data['vy'].to_numpy()
    particles.p[:, 2] = m * data['vz'].to_numpy()

    particles.angular_mom[:, 0] = particles.mom_inertia * data['omegax'].to_numpy()
    particles.angular_mom[:, 1] = particles.mom_inertia * data['omegay'].to_numpy()
    particles.angular_mom[:, 2] = particles.mom_inertia * data['omegaz'].to_numpy()

    particles.updateM()
    return particles

def dump_file_timestep(filename, basename='box'):
    ## Gets the integer value of the timestep from a dump filename.
    s = filename.strip(basename).strip('.dump')
    return int(s)


def load(filename):
    data = load_lammps_dump(filename)
    key = list(data.keys())[0]
    data = data[key]
    data['id'] = pd.to_numeric(data['id'], downcast='integer')
    data.sort_values('id', inplace=True)
    data.set_index('id', inplace=True)
    return data

def prepare_dataframe(N_particles, N_timesteps):
    columns = []
    for i in range(N_particles):
        columns.append('x' + str(i))
        columns.append('y' + str(i))
        columns.append('z' + str(i))
        columns.append('vx' + str(i))
        columns.append('vy' + str(i))
        columns.append('vz' + str(i))
        columns.append('omegax' + str(i))
        columns.append('omegay' + str(i))
        columns.append('omegaz' + str(i))
    index = np.arange(N_timesteps)
    df = pd.DataFrame(0.0, index=index, columns=columns)
    return df

def plot_trail(fig, x, z, trail_legend, particle_legend, label=None):

    tt = np.linspace(0, 2*np.pi, 101)
    xx = 0.5 * np.cos(tt)
    zz = 0.5 * np.sin(tt)

    #print(' ', particle_legend, x, z)
    fig.plot(x, z, trail_legend) #'C{0}-'.format(i))

    fig.plot(xx + x[-1],
                    zz + z[-1],
                    particle_legend)
    if not label is None:
        plt.text(x[-1], z[-1], label)



def make_cache_filename(**kwargs):
    return '{name}_{T}_{dt}_{k}_{gamma}_{N}.csv'.format(**kwargs)

def is_cache_fresh(cache_filename, dump_filename):
    if not os.path.exists(cache_filename):
        return False
    return os.path.getmtime(cache_filename) > os.path.getmtime(dump_filename)

def get_lammps_dump_files(dt, dump_folder, sort_key=dump_file_timestep):

    ## Load lammps data
    print(dump_folder)
    if os.path.exists(dump_folder):
        dump_files = os.listdir(dump_folder)
        dump_files.sort(key=sort_key)

        return dump_files
    return []

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

def calc_gran_temp(m, d, N_particles, data):
    data['ave_vx'] = np.zeros(data['t'].size)
    data['ave_vy'] = np.zeros(data['t'].size)
    data['ave_vz'] = np.zeros(data['t'].size)

    data['ave_omegax'] = np.zeros(data['t'].size)
    data['ave_omegay'] = np.zeros(data['t'].size)
    data['ave_omegaz'] = np.zeros(data['t'].size)

    for i in range(N_particles):
        data['ave_vx'] += data['vx' + str(i)]/N_particles
        data['ave_vy'] += data['vy' + str(i)]/N_particles
        data['ave_vz'] += data['vz' + str(i)]/N_particles
        data['ave_omegax'] += data['omegax' + str(i)]/N_particles
        data['ave_omegay'] += data['omegay' + str(i)]/N_particles
        data['ave_omegaz'] += data['omegaz' + str(i)]/N_particles

    data['T_xx'] = np.zeros(data['t'].size)
    data['T_yy'] = np.zeros(data['t'].size)
    data['T_zz'] = np.zeros(data['t'].size)
    for i in range(N_particles):
        data['T_xx'] += (data['ave_vx'] - data['vx' + str(i)]/N_particles)**2/N_particles
        data['T_xx'] += (data['ave_omegax'] - data['omegax' + str(i)]/N_particles)**2/N_particles

        data['T_yy'] += (data['ave_vy'] - data['vy' + str(i)]/N_particles)**2/N_particles
        data['T_yy'] += (data['ave_omegay'] - data['omegay' + str(i)]/N_particles)**2/N_particles

        data['T_zz'] += (data['ave_vz'] - data['vz' + str(i)]/N_particles)**2/N_particles
        data['T_zz'] += (data['ave_omegaz'] - data['omegaz' + str(i)]/N_particles)**2/N_particles

    data['gran_temp'] = (1/3) * (
        data['T_xx'] +
        data['T_yy'] +
        data['T_zz']
    )

def plot_lammps(N, d, dt, T, particles_to_plot, dump_folder, plot_xy, plot_zt, plot_K, plot_gran_temp, k, gamma, T_macro_scale, v_scale, K_scale):

    dump_files = get_lammps_dump_files(dt, dump_folder)
    # Starting configuration
    #data = load_lammps_dump('lammps/box/dump/' + dump_files[0])
    #data = data[1000]
    #data.sort_values('id', inplace=True)
    if dump_files == []:
        return

    T0 = dt * dump_file_timestep(dump_files[0])

    dump_files = [
        dump_file
        for dump_file in dump_files
        if dump_file_timestep(dump_file) <= (T + T0)/dt + 0.5
    ]
    N_particles = 0
    lammps_data = None

    lammps_cache_filename = make_cache_filename(
        name='cache/box_lammps',
        T=T,
        dt=dt,
        k=k,
        gamma=gamma,
        N=N,
    )
    is_lammps_cache_fresh = len(dump_files) > 0 and is_cache_fresh(lammps_cache_filename, dump_folder + dump_files[0])

    if os.path.exists(lammps_cache_filename):
        if is_lammps_cache_fresh:
            print('Found LAMMPS cache: {0}'.format(lammps_cache_filename))
        else:
            print('Found stale LAMMPS cache: {0}'.format(lammps_cache_filename))

    if (os.path.exists(lammps_cache_filename) and not '--no-lammps-cache' in sys.argv and is_lammps_cache_fresh):

        lammps_data = pd.read_csv(lammps_cache_filename)

        data = load(dump_folder + dump_files[0])
        m = data['mass'].to_numpy()[0]
        d = data['diameter'].to_numpy()[0]
        N_particles = data.shape[0]
    else:
        #print(dump_files)
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

    calc_gran_temp(m, d, N_particles, lammps_data)
    plot_gran_temp.plot(
        lammps_data['t']*T_macro_scale,
        lammps_data['gran_temp']/v_scale,
        'C{0}--'.format(plot_gran_temp.n_plot),
        label='LAMMPS ($\Delta t = {0}$)'.format(dt)
    )


    calc_total_kinetic_E(m, d, N_particles, lammps_data)
    plot_K.plot(
        lammps_data['t']*T_macro_scale,
        (lammps_data['K_lin'] + lammps_data['K_rot'])/(K_scale*N_particles),
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
            label=str(part_i+1))

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


def get_particles(figure_data, particles, t_step):
    for i in range(particles.xyz.shape[0]):
        particles.xyz[i, 0] = figure_data['x' + str(i)][t_step]
        particles.xyz[i, 1] = figure_data['y' + str(i)][t_step]
        particles.xyz[i, 2] = figure_data['z' + str(i)][t_step]
        particles.v[i, 0] = figure_data['vx' + str(i)][t_step]
        particles.v[i, 1] = figure_data['vy' + str(i)][t_step]
        particles.v[i, 2] = figure_data['vz' + str(i)][t_step]
    particles.is_rep = np.zeros(particles.xyz.shape[0], dtype=bool)



def main():

    #d = 0.25
    g = 9.8
    d = 1.
    #if '--d' in sys.argv:
    #    d = float(sys.argv[sys.argv.index('--d') + 1])
    ic(d)

    N = 200
    if '--N' in sys.argv:
        N = int(sys.argv[sys.argv.index('--N') + 1])
    

    L = 3
    if N == 400:
        L = 3.78
    elif N == 800:
        L = 4.74
    ic(L)

    L_zhi = 50
    k = 1000000
    rho = 1.000#*(2*np.sqrt(2))
    volume = 4./3. * np.pi * (d/2)**3
    m =  volume * rho

    gamma = 100
    #gamma_n = gamma * m
    #gamma_t = gamma_n/2

    dt = 0.0001
    if '--dt' in sys.argv:
        dt = float(sys.argv[sys.argv.index('--dt') + 1])
        
    ic(dt)
    T_scale = np.pi/np.sqrt(2*k/m)
    print('dt/t_c =', dt/T_scale)
    if '--profile' in sys.argv:
        T = 1000 * dt #0.5
    else:
        T = 5.0
        if '--T' in sys.argv:
            T = float(sys.argv[sys.argv.index('--T') + 1])
        #T = 1000 * dt #0.5
    ic(T)

    dt_dump = 0.001
    dump_step = int(dt_dump/dt+0.5)
    #dt_dump = dump_step*dt

    T_macro_scale = np.sqrt(g/d)
    v_scale = np.sqrt(g*d)
    K_scale = m*g*d

    # Figures
    particles_to_plot = [0, 1, 14, 20] #, 777]
    fig_pos_xy = Figure('x', 'Position (y)', show=True)
    fig_K = Figure(
        '$\sqrt{g/d}\ t$',
        r'$\bar{K}/(mgd)$',
        dat_filename='figures/box_K_{0}.dat',
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/box_K.tex',
    )
    fig_xt = Figure('t', 'x', show=False)
    fig_zt = Figure('t', 'z')
    fig_gran_temp = Figure(
        #'$t/t_c$',
        r'$\sqrt{g/d}\ t$',
        r'$\delta v/\sqrt{gd}$',
        dat_filename='figures/box_T_{0}.dat',
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/box_T.tex',
    )


    # Plot LAMMPS data
    T_lammps = 5
    lammps_dts = [0.0005, 0.0001]
    if not '--profile' in sys.argv:
        #for lammps_dt in [0.0005, 0.0001, 0.00001, 0.000001, 0.0000001]:
        for lammps_dt in lammps_dts:
            dump_folder = 'lammps/box/dump_{N}_{dt:.10f}'.format(N=N, dt=lammps_dt).strip('0')+'/'
            plot_lammps(N, d, lammps_dt, T_lammps, particles_to_plot, dump_folder, fig_pos_xy, fig_zt, fig_K, fig_gran_temp, k, gamma, T_macro_scale, v_scale, K_scale)

    dump_folder = 'lammps/box/dump_{N}_{dt:.10f}'.format(N=N, dt=dt).strip('0')+'/'
    cache_filename = make_cache_filename(name='cache/box_particles', T=T, dt=dt, k=k, gamma=gamma, N=N)
    # Starting configuration
    dump_files = get_lammps_dump_files(dt, dump_folder)

    initial_config = 'lammps/box/box_generate_{N}.dump'.format(N=N)
    ic(initial_config)
    if not os.path.exists(initial_config):
        print('Could not load initial partile configuration from LAMMPS')
        exit(0)
    data = load(initial_config)

    particles = setup_particles_from_lammps(data, k=k, gamma=gamma)

    nlist = NList(d/2)
    contact = HookianContact(particles, dt)
    N_particles = particles.xyz.shape[0]
    mass = particles.m[0]


    # Number of time steps used in plotting the figure (based on the largest time step)
    N_t_steps = int(T/dt)+1
    N_dump_steps = int(T/dt_dump)+1
    print('N_dump_steps:', N_dump_steps)

    is_simulation_cache_fresh = is_cache_fresh(cache_filename, initial_config)

    if os.path.exists(cache_filename) and not '--no-cache' in sys.argv:
        print('Found box simulation cache: {0}'.format(cache_filename))
        print('dt =', dt)
        figure_data = pd.read_csv(cache_filename)

        #for i in range(N_particles):
        #    i_dump = N_dump_steps -1
        #    particles.xyz[i, 0] = figure_data['x' + str(i)][i_dump]
        #    particles.xyz[i, 1] = figure_data['y' + str(i)][i_dump]
        #    particles.xyz[i, 2] = figure_data['z' + str(i)][i_dump]
        #    particles.v[i, 0] = figure_data['vx' + str(i)][i_dump]
        #    particles.v[i, 1] = figure_data['vy' + str(i)][i_dump]
        #    particles.v[i, 2] = figure_data['vz' + str(i)][i_dump]
        #    #particles.v[i, 0] = figure_data['vx' + str(i)][i_dump]
        #    #particles.v[i, 1] = figure_data['vy' + str(i)][i_dump]
        #    #particles.v[i, 2] = figure_data['vz' + str(i)][i_dump]
    else:
        # Perform our DEM simulation

        figure_data = prepare_dataframe(N_particles, N_dump_steps)
        figure_data['t'] = np.linspace(0, T, N_dump_steps)

        ## Wall setup

        walls = [
            Wall(np.array([-L, 0, 0]), np.array([ 1, 0, 0]), k, gamma, gamma /2),
            Wall(np.array([ L, 0, 0]), np.array([-1, 0, 0]), k, gamma, gamma /2),
            Wall(np.array([0, -L, 0]), np.array([0,  1, 0]), k, gamma, gamma /2),
            Wall(np.array([0,  L, 0]), np.array([0, -1, 0]), k, gamma, gamma /2),
            Wall(np.array([0, 0, -L]), np.array([0, 0,  1]), k, gamma, gamma /2),
            Wall(np.array([0, 0,  L_zhi]), np.array([0, 0, -1]), k, gamma, gamma /2),
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
                gravity=np.array([0, 0, -g]),
                nlist=nlist,
                contact_law=contact,
                #dq_alpha=0.5
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

    compare_to_plot_i = lammps_dts.index(dt)
    calc_total_kinetic_E(m, d, N_particles, figure_data)
    fig_K.plot(
        figure_data['t']*T_macro_scale,
        (figure_data['K_lin'] + figure_data['K_rot'])/(K_scale*N_particles),
        'C{0}-'.format(compare_to_plot_i),
        label='Var. Int. ($\Delta t = {0}$)'.format(dt)
    )

    calc_gran_temp(m, d, N_particles, figure_data)
    fig_gran_temp.plot(
        figure_data['t']*T_macro_scale,
        figure_data['gran_temp']/v_scale,
        'C{0}-'.format(compare_to_plot_i),
        label='Var. Int ($\Delta t = {0}$)'.format(dt)
    )

    #for i in range(20):
    for i, part_i in enumerate(particles_to_plot):
        plot_trail(
            fig_pos_xy,
            figure_data['x'+str(part_i)].to_numpy(),
            figure_data['z'+str(part_i)].to_numpy(),
            'b--',
            'b-',
            label=str(part_i+1))

        plt.plot([L, L, -L, -L, L], [L_zhi, -L, -L, L_zhi, L_zhi], 'k:')
        plt.axis('equal')

        fig_zt.plot(figure_data['t'].to_numpy(),
                    figure_data['z'+str(part_i)].to_numpy(),
                    'C{0}-'.format(i))

        plt.text(figure_data['t'].to_numpy()[-1],
                 figure_data['z'+str(part_i)].to_numpy()[-1],
                 str(part_i+1))

    #for i, t_norm in enumerate([0.5, 0.55, 0.6]):
    #    t = t_norm * np.sqrt(g/d)
    #    timestep = int(t/dt_dump + 0.5)
    #    get_particles(figure_data, particles, timestep)
    #    save_tikz_cartoon(
    #        'figures/box{0}.tex'.format(i+1),
    #        particles,
    #        #drawvel=True,
    #        p_scale=np.max(particles.p[:]),
    #        y_coord_idx=2,
    #        extra_cmd=['\drawbox']
    #    )


    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
