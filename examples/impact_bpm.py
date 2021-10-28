import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tikzplotlib

sys.path.append('.')

from vi_dem.particles import (
    Particles,
    NList,
)
from vi_dem.variational_int import (
    HookianContact,
    BondContact,
    SumContact,
    time_step,
)

from utils import *
from impact import (
    draw_circle,
    draw_v,
)


def E_test_setup(d, k, m):
    particles = Particles()
    N = 3
    particles.xyz = np.zeros((N, 3))
    particles.xyz[0] = (-d/2 - d/50, -0.3, 0)
    #particles.xyz[1] = (-d/2 - d/50, -d, 0)
    particles.xyz[1] = (-d/2 - d/50-d, -0.3, 0)
    particles.xyz[2] = ( d/2 + d/50, 0.3, 0)
    #particles.xyz[0] = (-1.5*d/2, 0, 0)
    #particles.xyz[1] = (-1.5*d/2, -d, 0)
    #particles.xyz[2] = ( 1.5*d/2, 0, 0)
    particles.d = np.ones(N) * d

    particles.rot = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))

    v = np.zeros(particles.xyz.shape)
    v[0] = (1, 0, 0)
    v[1] = (1, 0, 0)
    v[2] = (-1, 0, 0)

    particles.m = np.ones(particles.xyz.shape)
    particles.k = k

    particles.m = np.ones(N) * m
    particles.p = np.zeros(v.shape)
    for i in range(N):
        particles.p[i] = v[i] * particles.m[i]
    print('v =', particles.v)
    print('m =', particles.m)

    particles.updateM()
    return particles

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
    T = 1.0

    ## Figure setup
    show_analytic = False
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

    t_label = '$t/t_c$'
    fig_pos_x = Figure(t_label, 'Position, x component', show=False)
    fig_pos_xy = Figure('', '', show=True)
    fig_pos_xy_init = Figure('', '', show=True)
    fig_f_x = Figure(
        t_label,
        '|F|/kd',
        show=True,
        dat_filename='figures/impact_bpm_F_{0}.dat',
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/impact_bpm_F.tex',
    )
    ## For logarithmic y axis
    #plt.yscale("log")
    #axes = plt.gca()
    #axes.set_ylim([10e-1,10e2])

    fig_v_x = Figure(t_label, 'Linear velocity, x', show=False)
    fig_v_y = Figure(t_label, 'Linear velocity, y', show=False)
    fig_Es = Figure(
        t_label,
        '$K_L(t)/K_L(0)$',
        #dat_filename='figures/impact_K_lin.dat',
        #template_filename='figures/impact.tex_template',
        #tikz_filename='figures/impact_K_lin.tex',
        show=False,
    )
    fig_Ers = Figure(
        t_label, '$K_R(t)/K_L(0)$',
        #dat_filename='figures/impact_K_rot.dat',
        #template_filename='figures/impact.tex_template',
        #tikz_filename='figures/impact_K_rot.tex',
        show=False,
    )
    fig_Etotal = Figure(t_label, 'Total kinetic Energy ($(K_L + K_R)/K(0)$)', show=False)
    fig_omega = Figure(t_label, 'Angular velocity ($\omega_z$)', show=False)

    E0 = E_kinetic(particles)
    print('E0 =', E0)

    ## Plot the initial configuration
    #fig_pos_xy_init.plot(xx+pos_x[0, 0], yy+pos_y[0, 0], 'k-')
    #fig_pos_xy_init.plot(xx+pos_x[1, 0], yy+pos_y[1, 0], 'k-')
    #fig_pos_xy_init.plot(xx+pos_x[2, 0], yy+pos_y[2, 0], 'k-')
    plt.figure(fig_pos_xy_init.fig.number)
    draw_v(pos0[0, 0], pos0[0, 1], p0[0, 0], p0[0, 1])
    draw_v(pos0[1, 0], pos0[1, 1], p0[1, 0], p0[1, 1])
    draw_v(pos0[2, 0], pos0[2, 1], p0[2, 0], p0[2, 1])

    draw_circle(d/2, fig_pos_xy_init, pos0[0, 0], pos0[0, 1], 'k-')
    draw_circle(d/2, fig_pos_xy_init, pos0[1, 0], pos0[1, 1], 'k-')
    draw_circle(d/2, fig_pos_xy_init, pos0[2, 0], pos0[2, 1], 'k-')


    fig_pos_xy_init.plot(pos0[0:2, 0], pos0[0:2, 1], 'C2--')
    plt.axis('equal')
    plt.xticks([], [])
    plt.yticks([], [])
    #plt.savefig('impact_before.png', dpi=300)
    #tikzplotlib.save('figures/impact_bpm_before.tikz')

    #dt = 0.001
    Ts = {
        0: 0.025,
        0.1: 0.035,
        0.3: 0.125
    }
    params_list = [
        #(0.1, 0.1, 0.00001),
        #(0.1, 0.1, 0.0001),
        ##(1, 0.1, 0.00001),
        #(10, 0.1, 0.000001),
        #(10, 0.1, 0.00001),
        (10, 0.3, 0.0001),
        (10, 0.3, 0.00001),
    ]
    gamma_n = 0
    for i, (k_mult, dy, dt) in enumerate(params_list):
        print('dt =', dt)

        #if i == 2:
        #    break

        dx = np.sqrt((d/2)**2 - dy**2)
        T_collision = -(pos0[2, 0] - dx)/(p0[2, 0]/m)
        print('dx =', dx)
        print('T_collision =', T_collision)
        # Number of time steps used in plotting the figure (based on the largest time step)
        T = 2*Ts[dy]
        #T_min = 0.95*T_collision #0.12
        T_min = T_collision - 0.5*T_scale #0.1217 #T
        T_max = T_collision + 4*T_scale #0.1217 #T
        print('T_min =', T_min)
        N_t_steps = int(T/dt + 0.5)+1
        print('N_t_steps =',N_t_steps)
        print('T =',T)
        print('dt =',dt)
        #ts = np.linspace(T_min, T_max, N_t_steps)
        ts = np.linspace(0, T, N_t_steps)
        Es = np.zeros(N_t_steps)
        Ers = np.zeros(N_t_steps)
        omegaz = np.zeros(N_t_steps)
        pos_x = np.zeros((3, N_t_steps))
        pos_y = np.zeros((3, N_t_steps))
        d_0 = np.zeros((3, N_t_steps))
        f_x = np.zeros((3, N_t_steps))
        f_y = np.zeros((3, N_t_steps))
        v_x = np.zeros(N_t_steps)
        v_y = np.zeros(N_t_steps)
        #data['t'] = ts

        ## Create the time domain for the figure
        min_idx = int(round(T_min/dt))
        print('min_idx =', min_idx)
        print('dt*min_idx =', (dt*min_idx-T_collision)/T_scale)
        max_idx = int(T_max/dt+1.5)
        idx = np.arange(min_idx, max_idx)
        t_plot = (ts[idx] - T_collision)/T_scale
        print('N timesteps:',T_scale/dt)

        #if i == 0:
        #    data['t'] = t_plot

        #label = 'LAMMPS ($\Delta t = {2}$, $\Delta y = {0}$, $\gamma={1}$)'.format(
        #    dy,
        #    gamma_n,
        #    dt,
        #)
        #fig_f_x.plot(
        #    t_plot,
        #    lgts_atom_one_data['fx'][min_idx:max_idx].to_numpy(),
        #    'C{0}--'.format(i),
        #    label=label)


        #fig_pos_x.plot(
        #    t_plot,
        #    lgts_atom_one_data['x'][min_idx:max_idx].to_numpy() - lgts_atom_one_data['diameter'][min_idx:max_idx].to_numpy()/2,
        #    'C{0}--'.format(i),
        #    label=label)

        #fig_v_x.plot(
        #    t_plot,
        #    lgts_atom_one_data['vx'][min_idx:max_idx].to_numpy(),
        #    'C{0}--'.format(i),
        #    label=label)

        #fig_v_y.plot(
        #    t_plot,
        #    lgts_atom_one_data['vy'][min_idx:max_idx].to_numpy(),
        #    'C{0}--'.format(i),
        #    label=label)

        #fig_Es.plot(
        #    t_plot,
        #    lgts_atom_one_data['KinEng'][min_idx:max_idx].to_numpy()/lgts_atom_one_data['KinEng'][0],
        #    'C{0}--'.format(i),
        #    label=label)

        #fig_Ers.plot(
        #    t_plot,
        #    lgts_atom_one_data['RotKinEng'][min_idx:max_idx].to_numpy()/lgts_atom_one_data['KinEng'][0],
        #    'C{0}--'.format(i),
        #    label=label)

        #fig_Etotal.plot(
        #    t_plot,
        #    (lgts_atom_one_data['KinEng'][min_idx:max_idx].to_numpy() +
        #     lgts_atom_one_data['RotKinEng'][min_idx:max_idx].to_numpy())/lgts_atom_one_data['KinEng'][0],
        #    'C{0}--'.format(i),
        #    label=label)

        #fig_omega.plot(
        #    t_plot,
        #    lgts_atom_one_data['omegaz'][min_idx:max_idx].to_numpy(),
        #    'C{0}--'.format(i),
        #    label=label)




        alphas = [0.5] #[0, 0.5]
        for alpha in alphas:
            # Restore the initial particle setup
            particles.xyz = pos0.copy()
            particles.p = p0.copy()
            particles.rot = rot0.copy()
            particles.angular_mom = angular_mom0.copy()
            particles.xyz[0, 1] = -dy
            #particles.xyz[1, 1] = -(d + dy)
            particles.xyz[1, 1] = -dy
            particles.xyz[2, 1] = dy
            #particles.p = p0.copy()
            particles.gamma_n = gamma_n
            particles.gamma_t = 0.5*particles.gamma_n
            ic(gamma_n)

            E0 = E_kinetic(particles)

            # Setup the particle neighbor list
            nlist = NList(d/2)
            bond_contact = BondContact(particles, dt)
            hookian_contact = HookianContact(particles, dt, bonds=bond_contact)
            bond_contact.k = k_mult*hookian_contact.k
            #bond_contact.gamma_n = 100 * particles.gamma_n
            #bond_contact.gamma_t = 100 * particles.gamma_t
            #bond_contact.gamma_n = 100 * m/2
            contact = SumContact(particles, hookian_contact, bond_contact)

            t = 0
            Es[0] = 1
            Ers[0] = 0
            pos_x[:, 0] = particles.xyz[:, 0]
            pos_y[:, 0] = particles.xyz[:, 1]
            v_x[0] = particles.v[0, 0]
            omegaz[1] = 0
            print('Es.shape =', Es.shape)
            print('T/dt =', T/dt)
            j = 0
            dr = np.zeros(pos0.shape)
            #contact.alpha = alpha
            while t <= T:
                #ts[i] = t
                #idx = np.where(np.abs(t - ts) < dt/2)[0]
                #if np.any(idx):
                # Save the kinetic energy for plotting if we are at a one of the
                # plotting time steps.
                Es[j] = E_kinetic(particles)/E0
                Ers[j] = E_kinetic_rot(particles)/E0
                omegaz[j] = particles.angular_mom[0, 2]/particles.mom_inertia[0]
                #print(Ers[idx[0]], E_kinetic_rot(particles))
                pos_x[:, j] = particles.xyz[:, 0]
                pos_y[:, j] = particles.xyz[:, 1]
                d_0[:, j] = np.sqrt(
                    (particles.xyz[0, 0] - particles.xyz[:, 0])**2 +
                    (particles.xyz[0, 1] - particles.xyz[:, 1])**2 +
                    (particles.xyz[0, 2] - particles.xyz[:, 2])**2
                )
                f_x[0, j] = hookian_contact.calc_force_ij(particles.xyz[0], particles.xyz[1], 0, 1, 0, d_0[1, j])
                f_x[1, j] = hookian_contact.calc_force_ij(particles.xyz[0], particles.xyz[2], 0, 2, 0, d_0[2, j])
                f_x[2, j] = bond_contact.calc_force_ij(particles.xyz[0], particles.xyz[1], 0, 1, 0, d_0[1, j])
                f_y[0, j] = hookian_contact.calc_force_ij(particles.xyz[0], particles.xyz[1], 0, 1, 1, d_0[1, j])
                f_y[1, j] = hookian_contact.calc_force_ij(particles.xyz[0], particles.xyz[2], 0, 2, 1, d_0[2, j])
                f_y[2, j] = bond_contact.calc_force_ij(particles.xyz[0], particles.xyz[1], 0, 1, 1, d_0[1, j])
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
            #fig_f_x.plot(t_plot, f_x[0, idx], 'C{0}-'.format(i*3+1), label='$F_x$ 0-1 (Hook)')
            params_label = r'($k_{B}/k=%s$, $h\approx t_c/%.1f$' % (
                #'%.0f'.format(k_mult) if k_mult > 1 else '1/%.0f'.format(1/k_mult),
                '%.0f' % (k_mult) if k_mult > 1 else '1/%.0f' % (1/k_mult),
                T_scale/dt,
            )
            ic(params_label)
            fig_f_x.plot(
                t_plot,
                np.sqrt(f_x[1, idx]**2 + f_y[1, idx]**2)/(k*d),
                #f_x[1, idx],
                'C{0}-'.format(i*2+1),
                label='$|F_{02}|$ (Hookian)' + params_label,
            )
            fig_f_x.plot(
                t_plot,
                np.sqrt(f_x[2, idx]**2 + f_y[2, idx]**2)/(k*d),
                #f_x[2, idx]**2,
                'C{0}-'.format(i*2+2),
                label='$|F_{01}|$ (Bond)' + params_label,
            )

            #fig_f_x.plot(t_plot, f_y[0, idx], 'C{0}--'.format(i*3+1), label='$F_y$ 0-1 (Hook)')
            #fig_f_x.plot(t_plot, f_y[1, idx], 'C{0}--'.format(i*2+1),
            #             label='$F_y$ 0-2 (Hook)' + params_label)
            #fig_f_x.plot(t_plot, f_y[2, idx], 'C{0}--'.format(i*2+2),
            #             label='$F_y$ 0-1 (Bond)' + params_label)

            fig_v_x.plot(t_plot, v_x[idx], line_style, label=label)
            fig_v_y.plot(t_plot, v_y[idx], line_style, label=label)
            fig_Es.plot(t_plot, Es[idx], line_style, label=label)
            #fig_Es.plot(t_plot, Es[idx], 'C{0}x'.format(i))
            fig_Ers.plot(t_plot, Ers[idx], line_style, label=label)
            fig_Etotal.plot(t_plot, Ers[idx] + Es[idx], line_style, label=label)

            fig_pos_x.plot(t_plot, d_0[1, idx], 'C1-')
            fig_pos_x.plot(t_plot, d_0[2, idx], 'C2-')
            #fig_pos_x.plot(t_plot, pos_x[0, idx], 'C1-', label=label)
            #fig_pos_x.plot(t_plot, pos_x[1, idx], 'C2-', label=label)
            #fig_pos_x.plot(t_plot, pos_x[2, idx], 'C3-', label=label)


            fig_pos_xy.plot(pos_x[0, 0:-1], pos_y[0, 0:-1], line_style)
            #fig_pos_xy.plot(xx+pos_x[0, -2], yy+pos_y[0, -2], 'k-')
            draw_circle(d/2, fig_pos_xy, pos_x[0, -2], pos_y[0, -2], 'k-')
            fig_pos_xy.plot(pos_x[1, 0:-1], pos_y[1, 0:-1], line_style)
            #fig_pos_xy.plot(xx+pos_x[1, -2], yy+pos_y[1, -2], 'k-')
            draw_circle(d/2, fig_pos_xy, pos_x[1, -2], pos_y[1, -2], 'k-')
            fig_pos_xy.plot(pos_x[2, 0:-1], pos_y[2, 0:-1], line_style)
            #fig_pos_xy.plot(xx+pos_x[2, -2], yy+pos_y[2, -2], 'k-')
            draw_circle(d/2, fig_pos_xy, pos_x[2, -2], pos_y[2, -2], 'k-')

            fig_pos_xy.plot(pos_x[0:2, -2], pos_y[0:2, -2], 'C2--')

            draw_v(pos_x[0, -2], pos_y[0, -2], particles.p[0, 0], particles.p[0, 1])
            draw_v(pos_x[1, -2], pos_y[1, -2], particles.p[1, 0], particles.p[1, 1])
            draw_v(pos_x[2, -2], pos_y[2, -2], particles.p[2, 0], particles.p[2, 1])

            plt.axis('equal')
            plt.xticks([], [])
            plt.yticks([], [])

            #plt.savefig('impact_after.png', dpi=300)
            #tikzplotlib.save('figures/impact_bpm_after.tikz')
            #save_tikz_cartoon(
            #    'figures/impact_bpm_after.tex',
            #    particles,
            #    bonds=bond_contact.bonds,
            #    drawvel=True,
            #    p_scale=np.abs(p0[0,0]),
            #)
            #print('FINAL')
            #print('Q=')
            #for i_part in range(3):
            #    print(r'\coordinate (pos{0}) at ({1}, {2});'.format(
            #        i_part,
            #        particles.xyz[i_part, 0],
            #        particles.xyz[i_part, 1],
            #    ))
            #    print(r'\coordinate (p{0}) at ({1}, {2});'.format(
            #        i_part,
            #        d/4 * particles.p[i_part, 0] / np.abs(p0[0,0]),
            #        d/4 * particles.p[i_part, 1] / np.abs(p0[0,0]),
            #    ))

            #    N_steps_in_trail = 20
            #    idx = np.arange(0, N_t_steps, np.floor(N_t_steps/N_steps_in_trail), dtype=np.int)
            #    print(idx)
            #    print(r'\draw ')
            #    for i_idx in idx[:-1]:
            #        print(r'({0}, {1}) --'.format(
            #            pos_x[i_part, i_idx],
            #            pos_y[i_part, i_idx],
            #        ))
            #    print(r'({0}, {1});'.format(
            #        pos_x[i_part, idx[-1]],
            #        pos_y[i_part, idx[-1]],
            #    ))



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
                'k-',
                label=label,
            )

    ## Save .dat file
    #data.to_csv(filename, sep='\t', index=False)

    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
