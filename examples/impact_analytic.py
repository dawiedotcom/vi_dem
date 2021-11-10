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
    BondContact,
    time_step,
)

from utils import *


def E_test_setup(d, k, m):
    particles = Particles()
    N = 2
    particles.xyz = np.zeros((N, 3))
    particles.xyz[0] = (-d/2, 0, 0)
    particles.xyz[1] = ( d/2, 0, 0)

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

def E_test_setup2(d, k, m):
    particles = Particles()
    N = 2
    particles.xyz = np.zeros((N, 3))
    particles.xyz[0] = (-d/2 - d/1000, 0, 0)
    particles.xyz[1] = ( d/2 + d/1000, 0, 0)

    particles.rot = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))

    v = np.zeros(particles.xyz.shape)

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

def x_analytic2(t, m, d, v, k, gamma):
    t_c = 1/m * np.sqrt(2*k*m - gamma**2)
    x0 = d/1000
    C1 = x0 #v/t_c
    #C1 = m*v/(np.sqrt(2*k*m - gamma**2))
    print('C1 =', C1)
    omega = np.sqrt(k/(m/2))
    xi = gamma/(np.sqrt(2*k*m))
    return d/2 + C1 * np.exp(-t*gamma/(m)) * np.cos(t * t_c)

def v_analytic2(t, m, d, v, k, gamma):
    delta = 2*k*m - gamma**2
    t_c = 1/m * np.sqrt(delta)
    x0 = d/1000
    C1 = x0
    print('C1 =', C1)
    omega = np.sqrt(k/(m/2))
    xi = gamma/(np.sqrt(2*k*m))
    #return (C1*gamma/(2*)m * np.exp(-gamma*t/(m)) * np.sin(t * omega * np.sqrt(1 - xi**2))
    #        + omega*np.sqrt(1-xi**2)* C1 * np.exp(-t*gamma/(m)) * np.cos(t * omega * np.sqrt(1 - xi**2)))
    #t_c = omega * np.sqrt(1-xi**2)
    t_norm = t * t_c
    return C1*t_c*np.exp(-gamma*t/(m)) * (
        #gamma/np.sqrt(delta) * np.sin(t_norm)
        -np.sin(t_norm)
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
    show_analytic = True
    analytic_plotted = []
    T_collision = 0.0 #0.02
    T_min = 0 #0.0195
    T_max = T #0.022
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
    fig_trajectory = Figure('x', 'y', show=False)
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
        show=False,
    )
    fig_Ers = Figure(
        t_label, '$K_R(t)/K_L(0)$',
        dat_filename='figures/impact_K_rot%s_{0}.dat'% (fname_suffix),
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/impact_K_rot%s.tex'% (fname_suffix),
        show=False,
    )
    #fig_err_v_h = Figure(
    #    t_label, '$h$',
    #    dat_filename='figures/impact_analytic_err_v_h_{0}.dat',
    #    #template_filename='figures/impact.tex_template',
    #    tikz_filename='figures/impact_analytic_err_v_h.tex',
    #    show=True,
    #)

    fig_Etotal = Figure(t_label, 'Total kinetic Energy ($(K_L + K_R)/K(0)$)', show=False)
    fig_omega = Figure(t_label, 'Angular velocity ($\omega_z$)', show=False)
    fig_x_trunc_err = Figure(t_label, 'Truncation Error |x(t) - x_n|', show=True)
    fig_v_trunc_err = Figure(t_label, 'Truncation Error |v(t) - v_n|', show=True)

    E0 = E_kinetic(particles)
    print('E0 =', E0)

    #dt = 0.001
    Ts = {
        #0: 0.025, k==1e6
        0: 0.05,
        0.1: 0.035,
        0.3: 0.125
    }

    alphas = [0, 0.5]
    #params_list = [(gamma, dy, dt), ...]
    T_scale = np.pi/np.sqrt(2*k/m)
    T_scale2 = 1/np.sqrt(k/m)
    params_list = [
        #(100, 0.0, 0.000005),
        #(100, 0.0, 0.00001),
        #(100, 0.0, 0.000025),

        #(100, 0.0, 0.00005),
        #(100, 0.0, 0.0001),
        #(100, 0.0, 0.000125),
        #(100, 0.0, 0.00025),
        #(100, 0.0, 0.0003),

        #(100, 0.0, 0.0005),
        #(100, 0.0, 0.001),
        #(100, 0.0, 0.0025),
        #(100, 0.0, 0.005),

        #(100, 0.0, T_scale/1000),
        #(100, 0.0, T_scale/300),

        (0, 0.0, T_scale/100),
        (0, 0.0, T_scale/30),
        (0, 0.0, T_scale/10),
        (0, 0.0, T_scale/3),
    ]

    #dt1 = [T_scale/div for div in np.arange(60, 220, 20)]
    dts = (
        [T_scale2*dt_ for dt_ in np.linspace(0.01, 0.06, 5)] +
        [T_scale2*dt_ for dt_ in np.linspace(0.06, 0.2, 10)] +
        [T_scale2*dt_ for dt_ in np.linspace(0.2, 1.4, 20)]
    )
    dts.sort()
    params_list = [(0, 0.0, dt_) for dt_ in dts]
    #params_list = [
    #    (0, 0.0, T_scale/div)
    #    for div in
    #    np.arange(20, 200, 20)
    #]
    #params_list = [
    #    (0, 0.0, T_scale2*dt)
    #    for dt in
    #    #np.linspace(0.01, 1.5, 10)
    #    #np.linspace(0.01, 0.8, 30)
    #    np.linspace(0.01, 0.03, 11)
    #]
    dts = [dt for (_, _, dt) in params_list]
    cols = alphas.copy()
    err_x_df = pd.DataFrame(0.0, columns=cols, index=dts)
    err_v_df = pd.DataFrame(0.0, columns=cols, index=dts)
    #dt_alpha = 1.0
    for i, (gamma_n, dy, dt) in enumerate(params_list):
        ic(dt)
        T_scale = np.pi/np.sqrt(2*k/m)
        print('dt = T_scale/', T_scale/dt)

        #if i == 2:
        #    break

        dx = np.sqrt((d/2)**2 - dy**2)
        T_collision = 0 #-(pos0[1, 0] - dx)/(p0[1, 0]/m)
        ic('T_collision =', T_collision)
        # Number of time steps used in plotting the figure (based on the largest time step)
        T = 20 * T_scale #Ts[dy]
        T_min = 0 #T_collision
        T_max = T_collision + T #T_scale
        print('T_min =', T_min)
        N_t_steps = int(T/dt + 1.5)
        ic(N_t_steps)
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


        ## Create the time domain for the figure
        min_idx = int(round(T_min/dt))
        print('min_idx =', min_idx)
        print('dt*min_idx =', (dt*min_idx-T_collision)/T_scale)
        max_idx = int(T_max/dt+1.5)
        idx = np.arange(min_idx, max_idx)
        #t_plot = (ts[idx] - T_collision) /T_scale
        t_plot = np.zeros(N_t_steps)
        ic('N timesteps:',T_scale/dt)

        #if i == 0:
        #    data['t'] = t_plot


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
            #contact = HookianContact(particles, dt)
            contact = BondContact(particles, dt, alpha=alpha)

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



            while t <= T+dt:

                #ts[i] = t
                #idx = np.where(np.abs(t - ts) < dt/2)[0]
                #if np.any(idx):
                # Save the kinetic energy for plotting if we are at a one of the
                # plotting time steps.

                if j < Es.shape[0]:
                    t_plot[j] = t/T_scale
                    #Es[j] = E_kinetic(particles)/E0
                    #Ers[j] = E_kinetic_rot(particles)/E0
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

                #if t < T_scale + dt/2 and t > T_scale - dt/2:
                #    _, contact_stress = contact.calc_force_and_stiffness(
                #        particles.gen_coords,
                #        particles.d,
                #        nlist,
                #        particles.gen_M_matrix,
                #        dt,
                #    )

                j += 1
                t += dt
                #ic(t - t_plot[j]*T_scale)



            #print('v(T) =')
            #print(particles.v)
            ## Save the energy values to data buffer.
            ##dt_key = '{0:f}'.format(dt)
            ##data[dt_key] = (ts, Es)

            ### Display
            label = r'{3} order $h \approx t_c/{2:.1f}$, $\Delta y/d = {0}$, $\gamma={1}$'.format(
                dy,
                gamma_n,
                T_scale/dt,
                'First' if alpha == 0 else 'Second',
               )
            line_style = 'C{0}{1}'.format(
                min(i, 9),
                '-' if alpha == 0.5 else '-.'
            )
            fig_f_x.plot(t_plot, f_x[idx], line_style, label=label)
            fig_v_x.plot(t_plot, v_x[idx], line_style, label=label)
            #fig_v_y.plot(t_plot, v_y[idx], line_style, label=label)
            #fig_Es.plot(t_plot, Es[idx], line_style, label=label)
            ##fig_Es.plot(t_plot, Es[idx], 'C{0}x'.format(i))
            #fig_Ers.plot(t_plot, Ers[idx], line_style, label=label)
            #fig_Etotal.plot(t_plot, Ers[idx] + Es[idx], line_style, label=label)
            fig_pos_x.plot(t_plot, pos_x[idx], line_style, label=label)
            #fig_trajectory.plot(pos_x, pos_y, line_style, label=label)
            #fig_omega.plot(t_plot, omegaz[idx], line_style, label=label)

            ## Plot the analytic solutions
            if show_analytic and dy == 0: # and not gamma_n in analytic_plotted:

                t_compare_idx = np.where(t_plot > -dt/2)[0]
                t_compare = t_plot[t_compare_idx]

                analytic_plotted.append(gamma_n)
                label = r'Analytic $\gamma = {0}$, $\Delta y=0$'.format(gamma_n)
                plot_settings = {
                    'label': label,
                    'linestyle': '--',
                }
                t_collision = np.pi * 1/np.sqrt(k * 2 / m)
                #ts_analytic = np.linspace(0, t_collision, 201)
                tt = np.linspace(0, 1, 1001)

                x_an = -x_analytic(t_compare*T_scale, m, 1, 1, k, gamma_n*m/2)
                fig_pos_x.plot(t_compare, x_an, 'k', **plot_settings)
                #x_an = x_analytic(tt*T_scale, m, 1, 1, k, gamma_n*m/2)-d/2
                #fig_pos_x.plot(tt*T_scale, -0.5-x_an, 'k--', **plot_settings)

                v_an = -v_analytic(t_compare*T_scale, m, 1, 1, k, gamma_n*m/2)
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
                fig_x_trunc_err.plot(t_compare,  np.abs(x_an - pos_x[t_compare_idx]), line_style, label=label)
                fig_v_trunc_err.plot(t_compare,  np.abs(v_an - v_x[t_compare_idx]), line_style, label=label)

                err = np.linalg.norm( x_an - pos_x[t_compare_idx] )
                err_x_df[alpha][dt] = err

                err = np.linalg.norm( v_an - v_x[t_compare_idx] )
                err_v_df[alpha][dt] = err

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


    h_comp = err_x_df.index.to_numpy() * np.sqrt(k/m)
    ic(h_comp)

    fig_x_err_v_h = plt.figure()
    for alpha in alphas:
        #plt.loglog(err_x_df.index.to_numpy(), err_x_df[alpha].to_numpy(), 'k{0}'.format('-' if alpha == 0.5 else '--'))
        #plt.loglog(err_x_df.index.to_numpy(), err_x_df[alpha].to_numpy(), 'k{0}'.format('o' if alpha == 0.5 else '+'))
        #plt.loglog(Lh2, err_x_df[alpha].to_numpy(), 'k{0}'.format('-' if alpha == 0.5 else '--'))
        #plt.loglog(Lh2, err_x_df[alpha].to_numpy(), 'k{0}'.format('o' if alpha == 0.5 else '+'))
        plt.semilogy(h_comp, err_x_df[alpha].to_numpy(), 'k{0}'.format('-' if alpha == 0.5 else '--'))
        plt.semilogy(h_comp, err_x_df[alpha].to_numpy(), 'k{0}'.format('o' if alpha == 0.5 else '+'))
    #plt.xlabel('h')
    #plt.xlabel('$Lh^2$')
    plt.xlabel('$h\sqrt{k/m}$')
    plt.ylabel('||x(t_n)-x_n||')

    fig_v_err_v_h = plt.figure()
    for alpha in alphas:
        #plt.loglog(err_v_df.index.to_numpy(), err_v_df[alpha].to_numpy(), 'k{0}'.format('-' if alpha == 0.5 else '--'))
        #plt.loglog(err_v_df.index.to_numpy(), err_v_df[alpha].to_numpy(), 'k{0}'.format('o' if alpha == 0.5 else '+'))
        #plt.loglog(Lh2, err_v_df[alpha].to_numpy(), 'k{0}'.format('-' if alpha == 0.5 else '--'))
        #plt.loglog(Lh2, err_v_df[alpha].to_numpy(), 'k{0}'.format('o' if alpha == 0.5 else '+'))
        plt.semilogy(h_comp, err_v_df[alpha].to_numpy(), 'k{0}'.format('-' if alpha == 0.5 else '--'))
        plt.semilogy(h_comp, err_v_df[alpha].to_numpy(), 'k{0}'.format('o' if alpha == 0.5 else '+'))
    #plt.xlabel('h')
    #plt.xlabel('$Lh^2$')
    plt.xlabel('$h\sqrt{k/m}$')
    plt.ylabel('||v(t_n)-v_n||')


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
