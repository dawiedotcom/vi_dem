import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    time_step,
)

from utils import *


def E_test_setup(d, k, m):
    particles = Particles()
    N = 2
    particles.xyz = np.zeros((N, 3))
    particles.xyz[0] = (-d/2 - d/50, 0, 0)
    particles.xyz[1] = ( d/2 + d/50, 0, 0)

    particles.rot = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))

    v = np.zeros(particles.xyz.shape)
    v[0] = (1, 0, 0)
    v[1] = (-1, 0, 0)

    particles.m = np.ones(particles.xyz.shape[0])
    particles.d = np.ones(particles.xyz.shape[0])
    particles.k = k

    particles.m *= m
    particles.m *= d
    particles.p = np.zeros(v.shape) #* vstack(particles.m
    for i in range(v.shape[0]):
        particles.p[i] = v[i] * m
    print('v =', particles.v)
    print('m =', particles.m)

    particles.updateM()
    return particles

def get_lammps_data(dt, gamma_n=None, dy=None):
    filename = make_lammps_filename('lammps/impact_damped/impact', 'thermo', dt, gamma_n=gamma_n, dy=dy)
    return check_and_load_lammps(filename)

def get_lammps_atom_one(dt, gamma_n=None, dy=None):
    filename = make_lammps_filename('lammps/impact_damped/atom_one', 'dump', dt, gamma_n=gamma_n, dy=dy)
    return check_and_load_lammps(filename)


def x_analytic(t, m, d, v, k, gamma):
    t_c = 1/m * np.sqrt(2*k*m - gamma**2)
    C1 = v/t_c
    #C1 = m*v/(np.sqrt(2*k*m - gamma**2))
    print('C1 =', C1)
    omega = np.sqrt(k/(m/2))
    xi = gamma/(np.sqrt(2*k*m))
    return d/2 - C1 * np.exp(-t*gamma/(m)) * np.sin(t * t_c)

def v_analytic(t, m, d, v, k, gamma):
    delta = 2*k*m - gamma**2
    t_c = 1/m * np.sqrt(delta)
    C1 = v/np.sqrt(delta)
    print('C1 =', C1)
    omega = np.sqrt(k/(m/2))
    xi = gamma/(np.sqrt(2*k*m))
    #return (C1*gamma/(2*)m * np.exp(-gamma*t/(m)) * np.sin(t * omega * np.sqrt(1 - xi**2))
    #        + omega*np.sqrt(1-xi**2)* C1 * np.exp(-t*gamma/(m)) * np.cos(t * omega * np.sqrt(1 - xi**2)))
    #t_c = omega * np.sqrt(1-xi**2)
    t_norm = t * t_c
    return np.exp(-gamma*t/(m)) * (
        gamma*v/np.sqrt(delta) * np.sin(t_norm)
        -v * np.cos(t_norm)
    )

def draw_v(x, y, vx, vy):
    arrow_scale = 0.5
    plt.arrow(
        x, #pos_x[0, 0],
        y, #pos_y[0, 0],
        arrow_scale*vx, #*p0[0, 0],
        arrow_scale*vy, #*p0[0, 1],
        head_width=0.05,
        head_length=0.025,
    )

def draw_circle(r, fig, x, y, linespec):
    tt = np.linspace(0, 2*np.pi, 101)
    xx = r * np.cos(tt)
    yy = r * np.sin(tt)
    fig.plot(x+xx, y+yy, linespec)

def main():

    filename = 'figures/impact.dat'
    #d = 0.25
    d = 1.
    L = 4
    k = 1000000
    dx = 0.05 * d
    rho = 1.000#*(2*np.sqrt(2))
    volume = 4./3. * np.pi * (d/2)**3
    print('volume =', volume)
    m =  volume * rho


    print('m =', m)
    T = 0.04

    ## Figure setup
    show_analytic = '--analytic' in sys.argv #True
    analytic_plotted = []
    T_collision = 0.0 #0.02
    T_min = 0 #0.0195
    T_max = T #0.022
    T_scale = np.pi/np.sqrt(2*k/m)
    T_high = 1e-4
    T_low = 1e-6
    geom_steps = 1
    N = int(np.log10(T_high/T_low)*geom_steps + 1)
    dts = np.geomspace(T_low, T_high, N)

    def E_kinetic(particles):
        # Calculates the kinetic energy of the first particle
        return 0.5 * particles.m[0] * np.sum(particles.v[0, :] **2)
    def E_kinetic_rot(particles):
        return 0.5 * particles.mom_inertia[0] * np.sum(particles.angular_v[0, :]**2)

    # Figure filename
    filename = 'figures/impact.dat'
    # Figure data
    data = pd.DataFrame()

    #snapshot = Template('figures/dem_E_vs_time_snapshot.tex_template')

    # The initial position and velocity is saved and restored for each
    # time step
    dy = 0.0
    particles = E_test_setup(d, k, m)
    pos0 = particles.xyz.copy()
    p0 = particles.p.copy()
    rot0 = particles.rot.copy()
    angular_mom0 = particles.angular_mom.copy()
    print(pos0)
    print(p0)

    t_label = '$t/t_c$'
    fig_pos_x = Figure(t_label, 'Position, x component', show=True)
    fig_trajectory = Figure('x', 'y', show=True)
    fig_f_x = Figure(t_label, 'Inter particle force, x component', show=False)
    fig_v_x = Figure(t_label, 'Linear velocity, x', show=True)
    fig_v_y = Figure(t_label, 'Linear velocity, y', show=False)
    fname_suffix = ''
    if '--small-dt' in sys.argv:
        fname_suffix = '_smdt'
    if '--analytic' in sys.argv:
        fname_suffix = '_analytic'

    fig_Es = Figure(
        t_label,
        '$K_L(t)/K_L(0)$',
        dat_filename='figures/impact_K_lin%s_{0}.dat' % (fname_suffix),
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/impact_K_lin%s.tex' % (fname_suffix),
    )
    fig_Ers = Figure(
        t_label, '$K_R(t)/K_L(0)$',
        dat_filename='figures/impact_K_rot%s_{0}.dat'% (fname_suffix),
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/impact_K_rot%s.tex'% (fname_suffix),
        show=True,
    )
    fig_Etotal = Figure(t_label, 'Total kinetic Energy ($(K_L + K_R)/K(0)$)', show=False)
    fig_omega = Figure(t_label, 'Angular velocity ($\omega_z$)', show=False)

    E0 = E_kinetic(particles)
    print('E0 =', E0)

    #dt = 0.001
    Ts = {
        0: 0.025,
        0.1: 0.035,
        0.3: 0.125
    }

    if '--small-dt' in sys.argv:
        # Parameters for comparing contact models at small time steps.
        #alphas = [(0.0|0.5), ...]
        alphas = [0, 0.5]
        #params_list = [(gamma, dy, dt), ...]
        params_list = [
            (0, 0.3, 0.00001),
            (30, 0.1, 0.00001),
            #(30, 0.3, 0.0001),
            (100, 0.0, 0.00001),
            #(100, 0.1, 0.0001),
            #(0, 0.0, 0.0001, 0.5),
            #(30, 0.0, 0.00001, 1),
            #(30, 0.1, 0.00001, 1),
            #(0, 0.0, 0.0005, 0.3),
            #(0, 0.0, 0.0001, 0.3),
            #(0, 0.0, 0.00001),
            #(0, 0.1),
            #(30, 0.1),
            #(30, 0.1),
            #(30, 0.3),
        ]
    elif '--analytic' in sys.argv:
        alphas = [0.5]
        params_list = [
            (0, 0, 0.00001),
            (100, 0, 0.00001),
            (300, 0, 0.00001),
        ]
    else:
        # Parameters for testing accuracy of large time steps
        # and integration order
        alphas = [0, 0.5]
        params_list = [
            (30, 0.1, 0.0001),
            (30, 0.1, 0.0005),
            (30, 0.1, 0.001),
        ]
    #dt_alpha = 1.0
    for i, (gamma_n, dy, dt) in enumerate(params_list):
        print('dt =', dt)
        print('dt = T_scale/', T_scale/dt)

        #if i == 2:
        #    break

        dx = np.sqrt((d/2)**2 - dy**2)
        T_collision = -(pos0[1, 0] - dx)/(p0[1, 0]/m)
        print('T_collision =', T_collision)
        # Number of time steps used in plotting the figure (based on the largest time step)
        T = Ts[dy]
        T_min = T_collision #0.12
        if '--small-dt' in sys.argv:
            T_max = T_collision + T_scale #0.1217 #T
        else:
            T_max = T_collision + 1.5*T_scale #0.1217 #T
        print('T_min =', T_min)
        N_t_steps = int(T/dt + 1.5)
        ts = np.linspace(0, T, N_t_steps)
        Es = np.zeros(N_t_steps)
        Ers = np.zeros(N_t_steps)
        omegaz = np.zeros(N_t_steps)
        pos_x = np.zeros(N_t_steps)
        pos_y = np.zeros(N_t_steps)
        f_x = np.zeros(N_t_steps)
        f_y = np.zeros(N_t_steps)
        v_x = np.zeros(N_t_steps)
        v_y = np.zeros(N_t_steps)
        #data['t'] = ts

        lmps_data = get_lammps_data(dt, gamma_n=gamma_n, dy=dy)
        lmps_atom_one_data = get_lammps_atom_one(dt, gamma_n=gamma_n, dy=dy)
        if not (lmps_data is None and lmps_atom_one_data is None):
            lmps_data['t'] = dt * lmps_data['Step']
            lmps_atom_one_data['t'] = lmps_data['t']
            lmps_atom_one_data['KinEng'] = 0.5*m*(
                lmps_atom_one_data['vx']**2 +
                lmps_atom_one_data['vy']**2)
            lmps_atom_one_data['RotKinEng'] = 0.5*particles.mom_inertia[0]*lmps_atom_one_data['omegaz']**2


        ## Create the time domain for the figure
        min_idx = int(round(T_min/dt))
        print('min_idx =', min_idx)
        print('dt*min_idx =', (dt*min_idx-T_collision)/T_scale)
        max_idx = int(T_max/dt+0.5)
        idx = np.arange(min_idx, max_idx)
        t_plot = (ts[idx] - T_collision)/T_scale
        print('N timesteps:',T_scale/dt)

        #if i == 0:
        #    data['t'] = t_plot

        label = r'LAMMPS ($h \approx t_c/{2:.1f}$, $\Delta y = {0}$, $\gamma={1}$)'.format(
            dy,
            gamma_n,
            T_scale/dt,
        )
        #fig_f_x.plot(
        #    t_plot,
        #    lmps_atom_one_data['fx'][min_idx:max_idx].to_numpy(),
        #    'C{0}--'.format(i),
        #    label=label)


        #if not ('--small-dt' in sys.argv or '--analytic' in sys.argv) and not lmps_atom_one_data is None:
        if not lmps_atom_one_data is None:
            fig_pos_x.plot(
                t_plot,
                lmps_atom_one_data['x'][min_idx:max_idx].to_numpy() - lmps_atom_one_data['diameter'][min_idx:max_idx].to_numpy()/2,
                'C{0}--'.format(i),
                label=label)

            fig_v_x.plot(
                t_plot,
                lmps_atom_one_data['vx'][min_idx:max_idx].to_numpy(),
                'C{0}--'.format(i),
                label=label)

            #fig_v_y.plot(
            #    t_plot,
            #    lmps_atom_one_data['vy'][min_idx:max_idx].to_numpy(),
            #    'C{0}--'.format(i),
            #    label=label)

            fig_Es.plot(
                t_plot,
                lmps_atom_one_data['KinEng'][min_idx:max_idx].to_numpy()/lmps_atom_one_data['KinEng'][0],
                'C{0}--'.format(i),
                label=label)

            fig_Ers.plot(
                t_plot,
                lmps_atom_one_data['RotKinEng'][min_idx:max_idx].to_numpy()/lmps_atom_one_data['KinEng'][0],
                'C{0}--'.format(i),
                label=label)

            #fig_Etotal.plot(
            #    t_plot,
            #    (lmps_atom_one_data['KinEng'][min_idx:max_idx].to_numpy() +
            #     lmps_atom_one_data['RotKinEng'][min_idx:max_idx].to_numpy())/lmps_atom_one_data['KinEng'][0],
            #    'C{0}--'.format(i),
            #    label=label)

            #fig_omega.plot(
            #    t_plot,
            #    lmps_atom_one_data['omegaz'][min_idx:max_idx].to_numpy(),
            #    'C{0}--'.format(i),
            #    label=label)




        for alpha in alphas:
            # Restore the initial particle setup
            particles.xyz = pos0.copy()
            particles.p = p0.copy()
            particles.rot = rot0.copy()
            particles.angular_mom = angular_mom0.copy()
            particles.xyz[0, 1] = -dy
            particles.xyz[1, 1] = dy
            particles.gamma_n = gamma_n
            particles.gamma_t = 0.5*particles.gamma_n

            E0 = E_kinetic(particles)

            # Setup the particle neighbor list
            nlist = NList(d/2)
            contact = HookianContact(particles, dt)

            t = 0
            Es[0] = 1
            Ers[0] = 0
            pos_x[0] = particles.xyz[0, 0]
            pos_y[0] = particles.xyz[0, 1]
            v_x[0] = particles.v[0, 0]
            omegaz[1] = 0
            print('pos(0) =')
            print(particles.xyz)
            print('Es.shape =', Es.shape)
            print('T/dt =', T/dt)
            print('T_scale/dt =', T_scale/dt)
            j = 0
            dr = np.zeros(pos0.shape)
            contact.alpha = alpha
            while t <= T:

                #ts[i] = t
                #idx = np.where(np.abs(t - ts) < dt/2)[0]
                #if np.any(idx):
                # Save the kinetic energy for plotting if we are at a one of the
                # plotting time steps.

                if j < Es.shape[0]:
                    Es[j] = E_kinetic(particles)/E0
                    Ers[j] = E_kinetic_rot(particles)/E0
                    omegaz[j] = particles.angular_mom[0, 2]/particles.mom_inertia[0]
                    #print(Ers[idx[0]], E_kinetic_rot(particles))
                    pos_x[j] = particles.xyz[0, 0]# - d/2
                    pos_y[j] = particles.xyz[0, 1]# - d/2
                    f_x[j] = max(k*(d - (particles.xyz[0, 0] - particles.xyz[1, 0])), 0)
                    v_x[j] = particles.v[0, 0]
                    v_y[j] = particles.v[0, 1]

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



            print('v(T) =')
            print(particles.v)
            # Save the energy values to data buffer.
            #dt_key = '{0:f}'.format(dt)
            #data[dt_key] = (ts, Es)

            ## Display
            label = r'{3} order $h \approx t_c/{2:.1f}$, $\Delta y/d = {0}$, $\gamma={1}$'.format(
                dy,
                gamma_n,
                T_scale/dt,
                'First' if alpha == 0 else 'Second',
               )
            line_style = 'C{0}{1}'.format(
                i,
                '-.' if alpha == 0 else '-'
            )
            fig_f_x.plot(t_plot, f_x[idx], line_style, label=label)
            fig_v_x.plot(t_plot, v_x[idx], line_style, label=label)
            fig_v_y.plot(t_plot, v_y[idx], line_style, label=label)
            fig_Es.plot(t_plot, Es[idx], line_style, label=label)
            #fig_Es.plot(t_plot, Es[idx], 'C{0}x'.format(i))
            fig_Ers.plot(t_plot, Ers[idx], line_style, label=label)
            fig_Etotal.plot(t_plot, Ers[idx] + Es[idx], line_style, label=label)
            fig_pos_x.plot(t_plot, pos_x[idx], line_style, label=label)
            fig_trajectory.plot(pos_x, pos_y, line_style, label=label)
            fig_omega.plot(t_plot, omegaz[idx], line_style, label=label)

        #draw = []
        #for pos in particles.xyz:
        #    draw.append(r'\draw[] ({0}, {1}) circle ({2})'.format(
        #        pos[0],
        #        pos[1],
        #        particles.d
        #    ))

        #for i in range(particles.xyz.shape[0]):
        #    particles.xyz[i, 2] = i+1
        #tex = snapshot.render(
        #    positions=particles.xyz,
        #    r=particles.d/2,
        #    labeltext='$t={0}$; $h={1}$'.format(T, dt),
        #)
        #snapshot_out_filename = 'figures/dem_E_vs_time_end_dt_{0:f}.tex'.format(dt)
        #with open(snapshot_out_filename, 'w') as out:
        #    out.write(tex)


        ## Plot the analytic solutions
        if show_analytic and dy == 0 and not gamma_n in analytic_plotted:
            analytic_plotted.append(gamma_n)
            label = r'Analytic $\gamma = {0}$, $\Delta y=0$'.format(gamma_n)
            plot_settings = {
                'label': label,
                'linestyle': '-.',
            }
            t_collision = np.pi * 1/np.sqrt(k * 2 / m)
            #ts_analytic = np.linspace(0, t_collision, 201)
            tt = np.linspace(0, 1, 1001)

            #x_an = x_analytic(t_plot*T_scale, m, 1, 1, k, gamma_n*m/2)-d/2
            x_an = x_analytic(tt*T_scale, m, 1, 1, k, gamma_n*m/2)-d/2
            #fig_pos_x.plot(t_plot, x_an, 'C{0}'.format(i), **plot_settings)

            Es_analytic = 0.5 * m * v_analytic(tt*T_scale, m, 1, 1, k, gamma_n*m/2)**2
            fig_Es.plot(
                tt,
                Es_analytic/Es_analytic[0],
                #'C{0}-.'.format(i),
                'k--',
                label=label,
            )

    if fig_trajectory.show:
        plt.figure(fig_trajectory.fig.number)
        plt.axis('equal')
        #fig_trajectory.fig.axis('equal')

    ## Save .dat file
    #data.to_csv(filename, sep='\t', index=False)

    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
