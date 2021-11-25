import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

from icecream import (
    ic,
    install,
)
install()

sys.path.append('.')

from vi_dem.particles import (
    Particles,
    NList,
)
from vi_dem.variational_int import (
    HookianContact,
    BondContact,
    time_step,
)


from impact import (
    get_lammps_atom_one,
    get_lammps_data,
)
from utils import *


def E_test_setup(props, v0):
    particles = Particles()
    N = 2
    particles.xyz = np.zeros((N, 3))
    particles.xyz[0] = (-props.d/2, 0, 0)
    particles.xyz[1] = ( props.d/2, 0, 0)

    particles.rot = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))

    v = np.zeros(particles.xyz.shape)
    v[0] = (v0, 0, 0)
    v[1] = (-v0, 0, 0)

    particles.m = np.ones(particles.xyz.shape[0])
    particles.d = np.ones(particles.xyz.shape[0])
    particles.k = props.k

    particles.m *= props.m
    particles.p = np.zeros(v.shape) #* vstack(particles.m
    for i in range(v.shape[0]):
        particles.p[i] = v[i] * props.m
    print('v =', particles.v)
    print('m =', particles.m)

    particles.updateM()
    return particles

def x_analytic(t, props, v0):
    gamma = props.gamma * props.m / 2
    t_c = 1/props.m * np.sqrt(2*props.k*props.m - gamma**2)
    C1 = v0/t_c
    return props.d/2 - C1 * np.exp(-t*gamma/(props.m)) * np.sin(t * t_c)

def v_analytic(t, props, v0):
    gamma = props.gamma * props.m / 2
    delta = 2*props.k*props.m - gamma**2
    t_c = 1/props.m * np.sqrt(delta)
    t_norm = t * t_c
    return np.exp(-gamma*t/(props.m)) * (
        gamma*v0/np.sqrt(delta) * np.sin(t_norm)
        -v0 * np.cos(t_norm)
    )

def plot_lammps_data(x_fig, v_fig, err_x_fig, err_v_fig, err_e_fig, dir_name, particle_properties, v0, color=2, mark='s', label='LAMMPS', dts_only=None):
    files = [fname
        for fname in os.listdir(dir_name)
        if re.match('atom_one.*\.dump', fname)
    ]
    lammps_dts = [
        float(filename.split('_')[2])
        for filename in files
    ]

    columns = ['x_err', 'v_err', 'e_err']
    if dts_only is None:
        err_data = pd.DataFrame(0.0, columns=columns, index=lammps_dts)
    else:
        err_data = pd.DataFrame(0.0, columns=columns, index=dts_only)

    T_scale2 = 1/np.sqrt(particle_properties.k/particle_properties.m)
    T_scale = np.pi*T_scale2/np.sqrt(2)
    for dt, filename in zip(lammps_dts, files):
        if not dts_only is None and not np.any(np.abs(np.array(dts_only) - dt) < 1e-6):
            print('LAMMPS: Skipping ', filename)
            continue

        atom_one_data = check_and_load_lammps(os.path.join(dir_name, filename))
        tt = dt * np.arange(atom_one_data['x'].size)

        pos_vel_label = '{0}: h/t_c={1}'.format(label, dt/T_scale)
        plt.figure(x_fig.fig.number)
        plt.plot(tt/T_scale, atom_one_data['x'].to_numpy(), '--', label=pos_vel_label)

        plt.figure(v_fig.fig.number)
        plt.plot(tt/T_scale, atom_one_data['vx'].to_numpy(), '--', label=pos_vel_label)

        x_an = -x_analytic(tt, particle_properties, v0)
        v_an = -v_analytic(tt, particle_properties, v0)
        #plt.plot(tt, x_an, 'k--')
        e_an = (
            particle_properties.k*(x_an - x_an[0])**2 +
            0.5*particle_properties.m*v_an**2
        )

        atom_one_data['E'] = (
            particle_properties.k*(atom_one_data['x'] - atom_one_data['x'][0])**2 +
            0.5*particle_properties.m*atom_one_data['vx']**2
        )

        err_data['x_err'][dt] = np.linalg.norm(x_an - atom_one_data['x'])
        err_data['v_err'][dt] = np.linalg.norm(v_an - atom_one_data['vx'])
        err_data['e_err'][dt] = np.linalg.norm(e_an - atom_one_data['E'])

    err_data.sort_index(inplace=True)
    #ic(err_data)
    err_x_fig.plot(err_data.index.to_numpy()/T_scale2, err_data['x_err'].to_numpy(), 'C{0}{1}'.format(color, mark), label=label)
    err_x_fig.plot(err_data.index.to_numpy()/T_scale2, err_data['x_err'].to_numpy(), 'C{0}-'.format(color))

    err_v_fig.plot(err_data.index.to_numpy()/T_scale2, err_data['v_err'].to_numpy(), 'C{0}{1}'.format(color, mark), label=label)
    err_v_fig.plot(err_data.index.to_numpy()/T_scale2, err_data['v_err'].to_numpy(), 'C{0}-'.format(color))

    err_e_fig.plot(err_data.index.to_numpy()/T_scale2, err_data['e_err'].to_numpy(), 'C{0}{1}'.format(color, mark), label=label)
    err_e_fig.plot(err_data.index.to_numpy()/T_scale2, err_data['e_err'].to_numpy(), 'C{0}-'.format(color))



def E_kinetic(particles):
    # Calculates the kinetic energy of the first particle
    return 0.5 * particles.m[0] * np.sum(particles.v[0, :] **2)
def E_kinetic_rot(particles):
    return 0.5 * particles.mom_inertia[0] * np.sum(particles.angular_v[0, :]**2)


@CacheMyDataFrame('cache/impact_analytic/run_simulation')
def run_simulation(dt, particle_properties, alpha, T, particles=None):
    columns = [
        't',
        'pos_x',
        'v_x'
    ]
    N_t_steps = round(T/dt)
    index = np.arange(N_t_steps)
    data = pd.DataFrame(0.0, index=index, columns=columns)

    # Setup the particle neighbor list
    nlist = NList(particle_properties.d/2)
    #contact = HookianContact(particles, dt)
    contact = BondContact(particles, dt, alpha=alpha)

    t = 0
    j = 0
    dr = np.zeros(particles.xyz.shape)

    T_scale2 = 1/np.sqrt(particle_properties.k/particle_properties.m)
    T_scale = np.pi*T_scale2/np.sqrt(2)

    while t <= T+dt:

        if j < N_t_steps:
            data['t'][j] = t/T_scale
            data['pos_x'][j] = particles.xyz[0, 0]
            data['v_x'][j] = particles.v[0, 0]

        nlist.time_step(particles, dr)
        # Run the simulation for one time step
        dr = time_step(
            particles,
            dt,
            nlist=nlist,
            contact_law=contact,
        )

        j += 1
        t += dt

    return data
    

def main():

    particle_properties = NewParticleProperties(
        d=1.0,
        k=1000000,
        rho=1.0,
        gamma=0,
    )
    v0 = 1.0

    ## Figure setup
    show_analytic = True
    analytic_plotted = []

    #snapshot = Template('figures/dem_E_vs_time_snapshot.tex_template')

    # The initial position and velocity is saved and restored for each
    # time step
    particles = E_test_setup(particle_properties, v0)
    particles0 = E_test_setup(particle_properties, v0)

    t_label = '$t/t_c$'
    fig_pos_x = Figure(t_label, 'Position, x component', show=True)
    fig_v_x = Figure(t_label, 'Linear velocity, x', show=True)
    #fname_suffix = ''
    #if '--small-dt' in sys.argv:
    #    fname_suffix = '_smdt'
    #if '--analytic' in sys.argv:
    #    fname_suffix = '_analytic'

    #fig_Es = Figure(
    #    t_label,
    #    '$K_L(t)/K_L(0)$',
    #    dat_filename='figures/impact_K_lin%s_{0}.dat' % (fname_suffix),
    #    template_filename='figures/impact.tex_template',
    #    tikz_filename='figures/impact_K_lin%s.tex' % (fname_suffix),
    #    show=False,
    #)
    #fig_Ers = Figure(
    #    t_label, '$K_R(t)/K_L(0)$',
    #    dat_filename='figures/impact_K_rot%s_{0}.dat'% (fname_suffix),
    #    template_filename='figures/impact.tex_template',
    #    tikz_filename='figures/impact_K_rot%s.tex'% (fname_suffix),
    #    show=False,
    #)

    fig_x_trunc_err = Figure(t_label, 'Truncation Error |x(t) - x_n|', show=True)
    fig_v_trunc_err = Figure(t_label, 'Truncation Error |v(t) - v_n|', show=True)
    fig_e_trunc_err = Figure(t_label, 'Truncation Error |E_n/E0 - 1|', show=True)

    alphas = [0, 0.5]
    T_scale2 = 1/np.sqrt(particle_properties.k/particle_properties.m)
    T_scale = np.pi*T_scale2/np.sqrt(2)

    dts = (
        [T_scale2*dt_ for dt_ in np.geomspace(0.01, 0.2, 15)] +
        [T_scale2*dt_ for dt_ in np.linspace(0.2, 1.4, 20)]
    )
    #dts = dts[0:-1:10]
    dts.sort()
    if '--show-dts' in sys.argv:
        ic(dts)
        exit(0)
    params_list = [(0, dt_) for dt_ in dts]
    dts = [dt for (_, dt) in params_list]
    cols = alphas.copy()
    err_x_df = pd.DataFrame(0.0, columns=cols, index=dts)
    err_v_df = pd.DataFrame(0.0, columns=cols, index=dts)
    err_e_df = pd.DataFrame(0.0, columns=cols, index=dts)

    for i, (gamma_n, dt) in enumerate(params_list):
        ic(dt)
        print('dt = T_scale/', T_scale/dt)

        # Time the simulation for
        T = 20 * T_scale

        for alpha in alphas:
            # Restore the initial particle setup
            particles.xyz = particles0.xyz.copy()
            particles.p = particles0.p.copy()
            particles.rot = particles0.rot.copy()
            particles.angular_mom = particles0.angular_mom.copy()
            particles.gamma_n = gamma_n
            particles.gamma_t = 0.5*particles.gamma_n

            sim_data = run_simulation(
                dt,
                particle_properties,
                alpha,
                T,
                particles=particles
            )

            ### Display
            label = r'{2} order $h \approx t_c/{1:.1f}$, $\gamma={0}$'.format(
                gamma_n,
                T_scale/dt,
                'First' if alpha == 0 else 'Second',
               )
            line_style = 'C{0}{1}'.format(
                min(i, 9),
                '-' if alpha == 0.5 else '-.'
            )

            t_plot = sim_data['t'].to_numpy()
            pos_x = sim_data['pos_x'].to_numpy()
            v_x = sim_data['v_x'].to_numpy()

            fig_v_x.plot(
                t_plot,
                sim_data['v_x'].to_numpy(),
                line_style,
                label=label,
            )
            fig_pos_x.plot(
                t_plot,
                pos_x,
                line_style,
                label=label,
            )

            ## Plot the analytic solutions
            if show_analytic: 

                #t_compare_idx = np.where(t_plot > -dt/2)[0]
                t_compare = t_plot #[t_compare_idx]

                analytic_plotted.append(gamma_n)
                label = r'Analytic $\gamma = {0}$, $\Delta y=0$'.format(gamma_n)
                plot_settings = {
                    'label': label,
                    'linestyle': '--',
                }

                x_an = -x_analytic(t_compare*T_scale, particle_properties, v0)
                fig_pos_x.plot(t_compare, x_an, 'k', **plot_settings)

                v_an = -v_analytic(t_compare*T_scale, particle_properties, v0)
                fig_v_x.plot(
                    t_compare,
                    v_an,
                    #'C{0}-.'.format(i),
                    'k',
                    label=label,
                )

                label = r'{0} order $h \approx t_c/{1:.1f}$$'.format(
                    'First' if alpha == 0.0 else 'Second',
                    T_scale/dt,
                )
                fig_x_trunc_err.plot(t_compare,  (x_an - pos_x), line_style, label=label)
                fig_v_trunc_err.plot(t_compare,  (v_an - v_x), line_style, label=label)

                err = np.linalg.norm( x_an - pos_x )
                err_x_df[alpha][dt] = err

                err = np.linalg.norm( v_an - v_x )
                err_v_df[alpha][dt] = err

                v = particle_properties.k * (pos_x - particles0.xyz[0, 0])**2
                k = 0.5 * particle_properties.m * v_x**2 
                energy = k + v
                ic(v[:20])

                #E0 = 0.5 * particle_properties.m * v0 ** 2
                #err = energy/E0 - 1
                e_an = (
                    particle_properties.k * (x_an - x_an[0])**2 +
                    0.5 * particle_properties.m * v_an**2 
                )
                #err = e_an - e_an[0]
                #err = energy - e_an
                err = energy/energy[0] - 1
                #fig_e_trunc_err.plot(t_compare,  0.5 * particle_properties.m * v_an**2 + particle_properties.k * (x_an - particles0.xyz[0, 0])**2 - E0, line_style, label=label)
                #fig_e_trunc_err.plot(t_compare,  v - 0.5 * particle_properties.k * (x_an - particles0.xyz[0, 0])**2, line_style, label=label)
                #fig_e_trunc_err.plot(t_compare,  k - 0.5 * particle_properties.m * v_an**2, line_style, label=label)
                if alpha == 0.5:
                    fig_e_trunc_err.plot(t_compare,  err, line_style, label=label)
                err_e_df[alpha][dt] = np.linalg.norm( err )

    fig_x_err_v_h = Figure(
        '$h/\sqrt{k/m}$',
        '$||x(t_n) - x_n||$',
        dat_filename='figures/impact_analytic_errx_{0}.dat',
        template_filename='figures/semilogy.tex_template',
        tikz_filename='figures/impact_analytic_errx.tex',
    )
        
    fig_v_err_v_h = Figure(
        '$h/\sqrt{k/m}$',
        '$||v(t_n) - v_n||$',
        dat_filename='figures/impact_analytic_errv_{0}.dat',
        template_filename='figures/semilogy.tex_template',
        tikz_filename='figures/impact_analytic_errv.tex',
    )

    fig_e_err_v_h = Figure(
        '$h/\sqrt{k/m}$',
        '$||E(t_n)/E0 - 1||$',
        dat_filename='figures/impact_analytic_erre_{0}.dat',
        template_filename='figures/semilogy.tex_template',
        tikz_filename='figures/impact_analytic_erre.tex',
    )


    def do_plot(fig, data):
        h_comp = data.index.to_numpy()/T_scale2 #* np.sqrt(k/m)

        for alpha in alphas:
            label = 'VI ({0} order)'.format(
                'First' if alpha == 0.0 else 'Second'
            )
            fig.plot(
                h_comp,
                data[alpha].to_numpy(),
                'C0{0}'.format('--' if alpha == 0.0 else '-'),
            )
            fig.plot(
                h_comp,
                data[alpha].to_numpy(),
                'C0{0}'.format('+' if alpha == 0.0 else 'o'),
                label=label,
            )

            ax = plt.gca()
            ax.set_yscale('log')

    do_plot(fig_x_err_v_h, err_x_df)
    do_plot(fig_v_err_v_h, err_v_df)
    do_plot(fig_e_err_v_h, err_e_df)
    #for alpha in alphas:
    #    label = 'VI ({0} order)'.format(
    #        'First' if alpha == 0.0 else 'Second'
    #    )
    #    fig_v_err_v_h.plot(
    #        h_comp,
    #        err_v_df[alpha].to_numpy(),
    #        'C0{0}'.format('--' if alpha == 0.0 else '-')
    #    )
    #    fig_v_err_v_h.plot(
    #        h_comp,
    #        err_v_df[alpha].to_numpy(),
    #        'C0{0}'.format('+' if alpha == 0.0 else 'o'),
    #        label=label
    #    )

    #ax = plt.gca()
    #ax.set_yscale('log')

    plot_lammps_data(
        fig_pos_x,
        fig_v_x,
        fig_x_err_v_h,
        fig_v_err_v_h,
        fig_e_err_v_h,
        'lammps/impact_analytic/dump_verlet',
        #k, m, d, 1.0,
        particle_properties,
        v0,
        #dts_only=[dt for _,dt in params_list],
        color=1,
        mark='s',
        label='LAMMPS (velocity-Verlet)',
    )
    #plot_lammps_data(
    #    fig_pos_x,
    #    fig_v_x,
    #    fig_x_err_v_h,
    #    fig_v_err_v_h,
    #    'lammps/impact_analytic/dump_respa',
    #    #k, m, d, 1.0,
    #    particle_properties,
    #    v0,
    #    #dts_only=[dt for _,dt in params_list],
    #    color=2,
    #    mark='d',
    #    label='LAMMPS (rRESPA)',
    #)


    ## Save .dat file
    #data.to_csv(filename, sep='\t', index=False)

    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
