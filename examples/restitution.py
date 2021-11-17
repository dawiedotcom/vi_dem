import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from icecream import ic, install
install()

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

def restitution_test_setup(props):
    particles = Particles()
    N = 1
    particles.xyz = np.zeros((N, 3))
    particles.d = props.d * np.ones(N)

    v = np.zeros(particles.xyz.shape)
    v[0] = (0, -1, 0)

    particles.m = np.ones(particles.xyz.shape)
    particles.k = props.k
    #particles.gamma_n = gamma_n
    #particles.gamma_t = gamma_n/2


    particles.rot = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))
    particles.is_rep = np.zeros(N, bool)

    particles.m = props.m * np.ones(N)
    particles.p = v * particles.m
    ic(particles.v)
    ic(particles.m)

    particles.updateM()
    return particles

def get_lammps_data(dt, gamma_n=None, omegaz=None, theta=None):
    filename = make_lammps_filename('lammps/restitution/restitution', 'thermo', dt, gamma_n=gamma_n, dy=omegaz, theta=theta)
    return check_and_load_lammps(filename)

def get_lammps_atom_one(dt, gamma_n=None, omegaz=None, theta=None):
    filename = make_lammps_filename('lammps/restitution/atom_one', 'dump', dt, gamma_n=gamma_n, dy=omegaz, theta=theta)
    return check_and_load_lammps(filename)


def append_lammps_pot_e(lmp_atom_one_data, part_props, walls=[]):
   lmp_atom_one_data['PotEng'] = np.zeros(lmp_atom_one_data['x'].shape)
   for wall in walls:
       lmp_xyz = np.array([lmp_atom_one_data['x'],
                           lmp_atom_one_data['y'],
                           np.zeros(lmp_atom_one_data['x'].shape)]).transpose()
       dist_to = wall.dist_to(lmp_xyz)
       idx = np.where(part_props.d/2 - dist_to > 0)[0]
       lmp_atom_one_data['PotEng'][idx] += 0.5*part_props.k*(part_props.d/2 - dist_to[idx])**2



def append_lammps_kin_e(lmp_atom_one_data, part_props):
    lmp_atom_one_data['KinEngL'] = 0.5*part_props.m*(
        lmp_atom_one_data['vx']**2 +
        lmp_atom_one_data['vy']**2)
    #lmp_atom_one_data['momL'] = m*np.sqrt(
    #    lmp_atom_one_data['vx']**2 +
    #    lmp_atom_one_data['vy']**2)
    lmp_atom_one_data['KinEngR'] = 0.5*part_props.I*(
        lmp_atom_one_data['omegax']**2 +
        #lmp_atom_one_data['omegay']**2 +
        lmp_atom_one_data['omegaz']**2)
    lmp_atom_one_data['KinEng'] = (
        lmp_atom_one_data['KinEngL'] +
        lmp_atom_one_data['KinEngR']
    )



def plot_lammps_err_data(e_fig, err_E_fig, dir_name, particle_properties, color='b', label='LAMMPS', dts_only=None, walls=[]):
    files = [fname
        for fname in os.listdir(dir_name)
        if re.match('atom_one.*\.dump', fname)
    ]
    lammps_dts = [
        float(filename.split('_')[2])
        for filename in files
    ]

    columns = ['E_err']
    if dts_only is None:
        err_data = pd.DataFrame(0.0, columns=columns, index=lammps_dts)
    else:
        err_data = pd.DataFrame(0.0, columns=columns, index=dts_only)

    T_scale2 = 1/np.sqrt(particle_properties.k/particle_properties.m)
    T_scale = np.pi*T_scale2/np.sqrt(2)
    for dt, filename in zip(lammps_dts, files):
        if not dts_only is None and not np.any(np.abs(np.array(dts_only) - dt) < 1e-7):
            print('LAMMPS: Skipping ', filename)
            continue

        atom_one_data = check_and_load_lammps(os.path.join(dir_name, filename))

        append_lammps_kin_e(atom_one_data, particle_properties)
        append_lammps_pot_e(atom_one_data, particle_properties, walls=walls)

        atom_one_data['Eng'] = atom_one_data['KinEng'] + atom_one_data['PotEng']

        tt = dt * np.arange(atom_one_data['x'].size)

        lammps_E_err = atom_one_data['Eng']/atom_one_data['Eng'][0] - 1
        E_label = '{0}: h={1}'.format(label, dt)
        #plt.figure(e_fig.number)
        #plt.plot(tt/T_scale, lammps_E_err, '--', label=E_label)
        e_fig.plot(
            tt/T_scale,
            #np.abs(lammps_E_err),
            (lammps_E_err),
            'C{0}--'.format(min(9, lammps_dts.index(dt))),
            label=E_label,
        )


        #x_an = -x_analytic(tt, particle_properties, v0)
        #v_an = -v_analytic(tt, particle_properties, v0)
        ##plt.plot(tt, x_an, 'k--')
        err_data['E_err'][dt] = np.linalg.norm(lammps_E_err)


    ic(err_data)

    err_data.sort_index(inplace=True)
    #ic(err_data)
    plt.figure(err_E_fig.number)
    plt.semilogy(err_data.index.to_numpy()/T_scale2, err_data['E_err'].to_numpy(), '{0}o'.format(color))
    plt.semilogy(err_data.index.to_numpy()/T_scale2, err_data['E_err'].to_numpy(), '{0}-'.format(color), label=label)



def E_kinetic_lin(particles):
    return 0.5 * np.sum(np.sum((particles.p)**2, axis=1)/particles.m[:])

def E_kinetic_rot(particles):
    return 0.5 * np.sum(np.sum((particles.angular_mom)**2, axis=1)/particles.mom_inertia[:])


@CacheMyDataFrame('cache/restitution/run_simulation')
def run_simulation(dt, props, alpha, T, particles=None, walls=[]):

    T_scale = np.pi/np.sqrt(2*props.k/props.m)
    #N_t_steps = int((T_max - T_min)/dt)+1
    N_t_steps = round(T/dt)
    #ts = np.linspace(T_min, T_max, N_t_steps)
    columns = [
        't_plot',
        'Ekin',
        'EkinL',
        'EkinR',
        'Epot',
        'Etot',
        'pos_x',
        'pos_y',
        'f_x',
    ]
        
    data = pd.DataFrame(0.0, index=np.arange(N_t_steps), columns=columns)

    t = 0
    wall_nlist = WallNList(props.d/2, walls)
    contact = HookianContact(particles, dt, alpha=alpha)
    nlist = NList(props.d/2)
    dr = np.zeros(particles.xyz.shape)
    for j in range(N_t_steps):

        if j%1000 == 0:
            ic(j/N_t_steps, t)

        data['t_plot'][j] = t/T_scale
        data['pos_x'][j] = particles.xyz[0, 0]
        data['pos_y'][j] = particles.xyz[0, 1]
        data['EkinL'][j] = E_kinetic_lin(particles)
        data['EkinR'][j] = E_kinetic_rot(particles)
        for wall in walls:
            dist_to = wall.dist_to(particles.xyz)[0]
            if props.d/2-dist_to > 0:
                data['Epot'][j] += 0.5*props.k*(props.d/2-dist_to)**2
                #data['f_x'][j] += props.k*(props.d/2-dist_to)

        nlist.time_step(particles, dr)
        wall_nlist.time_step(particles, dr)

        # Run the simulation for one time step
        dr = time_step(
            particles,
            dt,
            walls=walls,
            contact_law=contact,
            nlist=nlist,
        )

        t += dt

    data['Ekin'] = data['EkinL'] + data['EkinR']
    data['Etot'] = data['Epot'] + data['Ekin']

    return data


def main():

    L = 1.01
    particle_properties = NewParticleProperties(
        d=1.0,
        k=1000000,
        rho=1.0,
        gamma=0,
    )

    def fround(t, dt=.00001):
        return int(t/dt)*dt

    #T_scale = np.pi/np.sqrt(2*particle_properties.k/particle_properties.m)
    T_scale2 = 1/np.sqrt(particle_properties.k/particle_properties.m)
    T_scale = np.pi*T_scale2/np.sqrt(2)
    T_free = (L-particle_properties.d)/1 # v will be set to 1 at t=0
    if '--scaling' in sys.argv:
        T = fround(180 * (T_scale ))#0.4
    else:
        T = fround(1800 * (T_scale ))#0.4
    print('T =', T)
    print('N_collisions=', (T)/(T_scale + T_free))

    walls = [
        Wall(np.array([0, -L/2, 0]), np.array([0, 1, 0]), particle_properties.k, 0, 0),
        Wall(np.array([0, L/2, 0]), np.array([0, -1, 0]), particle_properties.k, 0, 0),
    ]

    # The initial position and velocity is saved and restored for each
    # time step
    particles = restitution_test_setup(particle_properties)
    particles0 = restitution_test_setup(particle_properties)

    save_tikz_cartoon(
        'figures/restitution_before.tex',
        particles,
        drawvel=True,
        p_scale=np.abs(particles0.p[0,1]),
    )

    fig_Etot = Figure(
        '$t/t_c$',
        '$[K(t)+V(t)]/K(0) -1$',
        dat_filename='figures/restitution_E_{0}.dat',
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/restitution_E.tex'
    )
    #fig_omegaz = Figure('$t/t_c$', 'Angular Velocity ($\omega_z$)', show=False)

    if '--scaling' in sys.argv:
        dts = (
            [T_scale2*dt_ for dt_ in np.geomspace(0.01, 0.2, 15)] +
            [T_scale2*dt_ for dt_ in np.linspace(0.2, 1.4, 20)]
        )
    else:
        dts = [ 0.00001, 0.00005 ]

    if '--show-dts' in sys.argv:
        print("TIMESTEPS='{0}'".format(' '.join([str(dt) for dt in dts])))
        exit(0)
    #dts = dts[0:3]
    alphas = [0.0, 0.5]
    params_list = [
        (dt, 0.0, 0.0, 0.0, alpha)
        for dt in dts
        for alpha in alphas
    ]
    err_E_df = pd.DataFrame(0.0, columns=alphas, index=dts)

    for i, (dt, gamma, omegaz, theta, alpha) in enumerate(params_list):

        print('dt = t_c/',T_scale/dt)

        # Restore the initial particle setup
        particles.xyz = particles0.xyz.copy()
        p = particles0.p[0, 1]
        particles.p = particles0.p.copy()
        particles.p[0, 0] = p * np.sin(theta/180*3.14)
        particles.p[0, 1] = p * np.cos(theta/180*3.14)
        particles.rot = particles0.rot.copy()
        particles.angular_mom = particles0.angular_mom.copy()

        particles.gamma_n = gamma 
        particles.gamma_t = particles.gamma_n/2

        for wall in walls:
            wall.alpha = alpha
            wall.gamma_n = gamma 
            wall.gamma_t = gamma

        particles.angular_mom[0, 2] = omegaz * particles.mom_inertia[0]

        E0 = E_kinetic_lin(particles) + E_kinetic_rot(particles)

        # TODO: run_simulation
        sim_data = run_simulation(
            dt,
            particle_properties,
            alpha,
            T,
            particles=particles,
            walls=walls,
        )

        ## Display

        label = r'Var. Int. ($h={0}$, $\alpha={1}$)'.format(
            dt,
            '0' if alpha == 0 else '1/2',
        )

        E_err = sim_data['Etot']/E0-1

        if alpha == 0.5:
            fig_Etot.plot(
                sim_data['t_plot'],
                #np.abs(E_err),
                (E_err),
                'C{0}{1}'.format(
                    min(i//2, 9),
                    ':' if alpha == 0.0 else '-',
                ),
                label=label,
            )

        #err_E_df[alpha][dt] = np.sum(E_err)
        err_E_df[alpha][dt] = np.linalg.norm(E_err)


    err_E = plt.figure()
    for alpha in alphas:
        line = '--' if alpha == 0.0 else '-'
        mark = '+' if alpha == 0.0 else 'o'
        label = 'VI ({0} order)'.format('First' if alpha == 0.0 else 'Second')
        plt.semilogy(dts/T_scale2, err_E_df[alpha], 'k'+mark, label=label)
        plt.semilogy(dts/T_scale2, err_E_df[alpha], 'k'+line)

    plot_lammps_err_data(
        fig_Etot,
        err_E,
        'lammps/restitution/dump_verlet',
        particle_properties,
        walls=walls,
        dts_only=dts,
    )

    plt.xlabel('$h\sqrt{k/m}$')
    plt.ylabel('$|E_t(t_n)/E(0) - 1|$')
    plt.legend(loc='best')
    
    plt.figure(fig_Etot.fig.number)

    if not '--scaling' in sys.argv:
        #ax = plt.gca()
        #ax.set_yscale('log')
        T_min = 1650
        T_max = 1700
        _, _, ymax, ymin = plt.axis()
        plt.axis([T_min, T_max, ymax, ymin])

    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
