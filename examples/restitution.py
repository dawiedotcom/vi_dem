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

def restitution_test_setup(d, k, m):
    particles = Particles()
    N = 1
    particles.xyz = np.zeros((N, 3))
    particles.d = d * np.ones(N)

    v = np.zeros(particles.xyz.shape)
    v[0] = (0, -1, 0)

    particles.m = np.ones(particles.xyz.shape)
    particles.k = k
    #particles.gamma_n = gamma_n
    #particles.gamma_t = gamma_n/2


    particles.rot = np.zeros((N, 3))
    particles.angular_mom = np.zeros((N, 3))
    particles.is_rep = np.zeros(N, np.bool)

    particles.m = m * np.ones(N)
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

def x_analytic(t, m, d, v, k, gamma):
    t_c = 1/m * np.sqrt(k*m - gamma**2)
    C1 = v/t_c
    return C1 * np.exp(-t*gamma/(m)) * np.sin(t * t_c)

def v_analytic(t, m, d, v, k, gamma):
    t_c = 1/m * np.sqrt(k*m - gamma**2)
    C1 = v/t_c
    #C1 = m*v/(np.sqrt(2*k*m - gamma**2))
    #omega = np.sqrt(k/(m))
    #xi = gamma/(np.sqrt(2*k*m))
    #return C1 * np.exp(-t*gamma/(m)) * np.sin(t * t_c)
    t_norm = t * t_c
    return np.exp(-gamma*t/(m)) * (
        gamma*v/np.sqrt(t_c) * np.sin(t_norm)
        -v * np.cos(t_norm)
    )

def main():

    filename = 'figures/restitution.dat'
    #d = 0.25
    d = 1.
    L = 1.01
    k = 1000000
    dx = 0.05 * d
    rho = 1.000#*(2*np.sqrt(2))
    volume = 4./3. * np.pi * (d/2)**3
    m =  volume * rho
    mom_inertia = 2./5. * m * (d/2)**2

    g=9.8
    print('k/(mg/d) =', k/(m*g/d))

    gamma = 10
    #gamma_n = gamma * m
    #gamma_t = gamma_n/2

    omegaz = 0

    def fround(t, dt=.00001):
        return int(t/dt)*dt

    T_scale = np.pi/np.sqrt(2*k/m)
    T_free = (L-d)/1 # v will be set to 1 at t=0
    T = fround(1800 * (T_scale ))#0.4
    print('T =', T)
    print('N_collisions=', (T)/(T_scale + T_free))

    walls = [
        Wall(np.array([0, -L/2, 0]), np.array([0, 1, 0]), k, gamma * m, gamma * m/2),
        Wall(np.array([0, L/2, 0]), np.array([0, -1, 0]), k, gamma * m, gamma * m/2),
    ]

    ## Figure setup
    show_analytic = True
    analytic_plotted = []
    #T_collision = 0.0
    T_min = fround(1650*(T_scale)) #0.0
    T_max = fround(1700*(T_scale)) #T
    T_high = 1e-4
    T_low = 1e-6
    geom_steps = 1
    N = int(np.log10(T_high/T_low)*geom_steps + 1)
    dts = np.geomspace(T_low, T_high, N)

    def E_kinetic_lin(particles):
        return 0.5 * np.sum(np.sum((particles.p)**2, axis=1)/particles.m[:])

    def E_kinetic_rot(particles):
        return 0.5 * np.sum(np.sum((particles.angular_mom)**2, axis=1)/particles.mom_inertia[:])

    # Figure filename
    filename = 'figures/impact.dat'
    # Figure data
    data = pd.DataFrame()

    #snapshot = Template('figures/dem_E_vs_time_snapshot.tex_template')

    # The initial position and velocity is saved and restored for each
    # time step
    particles = restitution_test_setup(d, k, m)

    pos0 = particles.xyz.copy()
    p0 = particles.p.copy()
    rot0 = particles.rot.copy()
    angular_mom0 = particles.angular_mom.copy()

    save_tikz_cartoon(
        'figures/restitution_before.tex',
        particles,
        drawvel=True,
        p_scale=np.abs(p0[0,1]),
    )

    fig_pos_y = Figure('$t/t_c$', 'Position (y)', show=False)
    fig_pos_xy = Figure('x', 'Position (y)', show=False)
    fig_Ekin = Figure('$t/t_c$', 'Kinetic Energy', show=False)
    fig_mom = Figure('$t/t_c$', 'Momentum', show=False)
    fig_V = Figure('$t/t_c$', 'Potential Energy', show=False)
    fig_EkinL = Figure('$t/t_c$', 'Linear Kinetic Energy', show=False)
    fig_Etot = Figure(
        '$t/t_c$',
        '$[K(t)+V(t)]/K(0) -1$',
        dat_filename='figures/restitution_E_{0}.dat',
        template_filename='figures/impact.tex_template',
        tikz_filename='figures/restitution_E.tex'
    )
    fig_omegaz = Figure('$t/t_c$', 'Angular Velocity ($\omega_z$)', show=False)

    E0 = (E_kinetic_lin(particles) + E_kinetic_rot(particles))
    print('E0 =', E0)

    #dt = 0.00001
    params_list = [
        #(0.00001, 0, 0, 0),
        (0.00005, 0, 0, 0, 0.5),
        (0.00001, 0, 0, 0, 0.5),
        #(0.00005, 0, 0, 0, 0),
        #(10, 0, 0),
        #(10, 0, 0),
        #(30, 3.14, 0),
        #(30, 0, 10.),
        #(30, 3.14, 10.),
        #(g, omega)
        #for g in [10]
        ##for y in [0, 0.1, 0.3]
        #for omega in [0, 3.14]
    ]
    for i, (dt, gamma, omegaz, theta, alpha) in enumerate(params_list):

        print('dt = t_c/',T_scale/dt)

        # Number of time steps used in plotting the figure (based on the largest time step)
        N_t_steps = int((T_max - T_min)/dt)+1
        ts = np.linspace(T_min, T_max, N_t_steps)
        Ekin = np.zeros(N_t_steps)
        EkinL = np.zeros(N_t_steps)
        EkinR = np.zeros(N_t_steps)
        Epot = np.zeros(N_t_steps)
        sim_omegas = np.zeros(N_t_steps)
        pos_y = np.zeros(N_t_steps)
        pos_x = np.zeros(N_t_steps)
        f_x = np.zeros(N_t_steps)
        #data['t'] = ts
        min_idx = int(T_min/dt)
        max_idx = int(T_max/dt)

        lmp_data = get_lammps_data(dt, gamma_n=gamma, omegaz=omegaz, theta=theta)
        lmp_atom_one_data = get_lammps_atom_one(dt, gamma_n=gamma, omegaz=omegaz, theta=theta)
        if not lmp_data is None and not lmp_atom_one_data is None:
            lmp_data['t'] = dt * lmp_data['Step']
            lmp_atom_one_data['t'] = lmp_data['t']

            lmp_atom_one_data['KinEngL'] = 0.5*m*(
                lmp_atom_one_data['vx']**2 +
                lmp_atom_one_data['vy']**2)
            #lmp_atom_one_data['momL'] = m*np.sqrt(
            #    lmp_atom_one_data['vx']**2 +
            #    lmp_atom_one_data['vy']**2)
            lmp_atom_one_data['KinEngR'] = 0.5*mom_inertia*(
                lmp_atom_one_data['omegax']**2 +
                #lmp_atom_one_data['omegay']**2 +
                lmp_atom_one_data['omegaz']**2)
            lmp_atom_one_data['KinEng'] = (
                lmp_atom_one_data['KinEngL'] +
                lmp_atom_one_data['KinEngR']
            )

            #lmp_atom_one_data['PotEng'] = 0.5 * k * (lmp_atom_one_data['y']**2)
            lmp_atom_one_data['PotEng'] = np.zeros(lmp_atom_one_data['x'].shape)
            for wall in walls:
                lmp_xyz = np.array([lmp_atom_one_data['x'],
                                    lmp_atom_one_data['y'],
                                    np.zeros(lmp_atom_one_data['x'].shape)]).transpose()
                print('lmp_xyz.shap =', lmp_xyz.shape)
                dist_to = wall.dist_to(lmp_xyz)
                idx = np.where(d/2 - dist_to > 0)[0]
                print('idx.shape =', idx.shape)
                lmp_atom_one_data['PotEng'][idx] += 0.5*k*(d/2 - dist_to[idx])**2
            #print(lmp_data)

            label = r'LAMMPS ($\omega_z={0}$, $\gamma_n={1}$, $\theta={2}$)'.format(
                omegaz,
                gamma,
                theta)
            fig_pos_y.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                lmp_atom_one_data['y'][min_idx:max_idx],
                'C{0}--'.format(i),
                label=label)

            fig_pos_xy.plot(
                lmp_atom_one_data['x'][min_idx:max_idx],
                lmp_atom_one_data['y'][min_idx:max_idx],
                'C{0}--'.format(i),
                label=label)

            fig_EkinL.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                lmp_atom_one_data['KinEngL'][min_idx:max_idx]/lmp_atom_one_data['KinEng'][0],
                'C{0}--'.format(i),
                label=label)


            e_kin_an = 0.5*m*v_analytic(lmp_atom_one_data['t'][min_idx:max_idx], m, d, p0[0, 1]/m, k, gamma*m/2)**2
            fig_Ekin.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                (lmp_atom_one_data['KinEng'][min_idx:max_idx]-e_kin_an)/lmp_atom_one_data['KinEng'][0],
                'C{0}--'.format(i),
                label=label)

            e_pot_an = 0.5*k*x_analytic(lmp_atom_one_data['t'][min_idx:max_idx], m, d, p0[0, 1]/m, k, gamma*m/2)**2
            fig_V.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                (lmp_atom_one_data['PotEng'][min_idx:max_idx]-e_pot_an)/lmp_atom_one_data['KinEng'][0],
                'C{0}--'.format(i),
                label=label)


            fig_Etot.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                (lmp_atom_one_data['KinEng'][min_idx:max_idx] +
                 lmp_atom_one_data['PotEng'][min_idx:max_idx])/lmp_atom_one_data['KinEng'][0] -1,
                'C{0}--'.format(i),
                label=label)

            fig_mom.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                (m*lmp_atom_one_data['vx'][min_idx:max_idx]),
                'C{0}--'.format(2*i),
                label=label)


            fig_omegaz.plot(
                lmp_atom_one_data['t'][min_idx:max_idx]/T_scale,
                lmp_atom_one_data['omegaz'][min_idx:max_idx],
                'C{0}--'.format(i),
                label=label)


        # Restore the initial particle setup
        particles.xyz = pos0.copy()
        p = p0[0, 1]
        particles.p = p0.copy()
        print('p0 =', p0)
        particles.p[0, 0] = p * np.sin(theta/180*3.14)
        particles.p[0, 1] = p * np.cos(theta/180*3.14)
        particles.rot = rot0.copy()
        particles.angular_mom = angular_mom0.copy()

        particles.gamma_n = gamma * m/2
        particles.gamma_t = particles.gamma_n/2

        for wall in walls:
            wall.alpha = alpha
            wall.gamma_n = gamma * m
            wall.gamma_t = gamma * m/2
        wall_nlist = WallNList(d/2, walls)
        contact = HookianContact(particles, dt, alpha=alpha)
        nlist = NList(d/2)

        particles.angular_mom[0, 2] = omegaz * particles.mom_inertia[0]
        print('omega(0) =')
        print(particles.angular_v)

        E0 = E_kinetic_lin(particles) + E_kinetic_rot(particles)

        t = 0
        #Es[0] = 1
        #sim_omegas[0] = omegaz
        #pos_y[0] = particles.xyz[0, 1]
        #pos_x[0] = particles.xyz[0, 0]
        dr = np.zeros(pos0.shape)
        j=0
        while t <= T+10*dt:

            #ts[i] = t
            #idx = np.where(np.abs(t - ts) < dt/2)[0]
            #if np.any(idx):
            #    # Save the kinetic energy for plotting if we are at a one of the
            #    # plotting time steps.
            #    Es[idx[0]] = E_kinetic(particles)/E0
            #    pos_x[idx[0]] = particles.xyz[0, 0]
            #    pos_y[idx[0]] = particles.xyz[0, 1]
            #    sim_omegas[idx[0]] = particles.angular_v[0, 2]
            #if t >= T_min and t <= T_max:
            if j >= min_idx and j <= max_idx:
                j_ = j-min_idx
                EkinL[j_] = E_kinetic_lin(particles)
                EkinR[j_] = E_kinetic_rot(particles)
                Ekin[j_] = (E_kinetic_lin(particles) + E_kinetic_rot(particles))
                pos_x[j_] = particles.xyz[0, 0]
                pos_y[j_] = particles.xyz[0, 1]
                #Epot[j_] = 0.5*k*pos_y[j]**2
                for wall in walls:
                    dist_to = wall.dist_to(particles.xyz)[0]
                    if d/2-dist_to > 0:
                        Epot[j_] += 0.5*k*(d/2-dist_to)**2

                sim_omegas[j_] = particles.angular_v[0, 2]

            nlist.time_step(particles, dr)
            wall_nlist.time_step(particles, dr)
            # Run the simulation for one time step
            dr = time_step(particles, dt, walls=walls, contact_law=contact, nlist=nlist,)

            t += dt
            j += 1



        # Save the energy values to data buffer.
        #dt_key = '{0:f}'.format(dt)
        #data[dt_key] = (ts, Es)


        print('omega(T) =')
        print(particles.angular_v)
        print('v(T) =')
        print(particles.v)
        ## Display
        label = r'Var. Int. ($\omega_z = {0}$ s, $\gamma_n={1}$, $\theta={2}$, $\alpha={3}$)'.format(
            omegaz,
            gamma,
            theta,
            '0' if alpha == 0 else '1/2',
        )
        fig_EkinL.plot(ts/T_scale, EkinL/E0, 'C{0}-'.format(i), label=label)
        #fig_Ekin.plot(ts/T_scale, Ekin/E0, 'C{0}-'.format(i), label=label)
        e_kin_an = 0.5*m*v_analytic(ts, m, d, p0[0, 1]/m, k, gamma*m/2)**2
        e_pot_an = 0.5*k*x_analytic(ts, m, d, p0[0, 1]/m, k, gamma*m/2)**2
        fig_Ekin.plot(ts/T_scale, (Ekin - e_kin_an)/E0, 'C{0}-'.format(i), label=label)
        fig_Etot.plot(ts/T_scale, (Ekin+Epot)/E0-1, 'C{0}-'.format(i), label=label)
        fig_V.plot(ts/T_scale, (Epot - e_pot_an)/E0, 'C{0}-'.format(i), label=label)
        fig_pos_y.plot(ts/T_scale, pos_y, 'C{0}-'.format(i), label=label)
        fig_pos_xy.plot(pos_x, pos_y, 'C{0}-'.format(i), label=label)
        fig_omegaz.plot(ts/T_scale, sim_omegas, 'C{0}-'.format(i), label=label)


        ## Plot the analytic solutions
        if show_analytic and omegaz == 0 and not gamma in analytic_plotted:
            analytic_plotted.append(gamma)
            label = r'Analytic $\gamma = {0}$'.format(gamma)
            plot_settings = {
                'label': label,
            }
            #ts_analytic = np.linspace(0, t_collision, 201)
            #tt = np.linspace(0, 1, 1001)
            steps = int(T/dt)+1
            tt = np.linspace(0, T+dt, steps)
            idx = np.where((tt >= T_min) & (tt <= T_max))[0]
            #x_an = x_analytic(t_plot*T_scale, m, 1, 1, k, gamma_n*m/2)-d/2

            v0 = p0[0, 1]/m
            x_an = x_analytic(tt, m, d, v0, k, gamma*m/2)[idx]
            v_an = v_analytic(tt, m, d, v0, k, gamma*m/2)[idx]
            fig_pos_y.plot(tt[idx]/T_scale, x_an, 'k-', **plot_settings)

            #fig_EkinL.plot(tt[idx]/T_scale, v_an**2 / v0**2 , 'k-', **plot_settings)

            #fig_V.plot(tt[idx]/T_scale, (0.5 * k * x_an**2) / (0.5 * m * v0**2) , 'k-', **plot_settings)

            fig_Etot.plot(tt[idx]/T_scale, (0.5 * k * x_an**2 + 0.5 * m * v_an**2) / (0.5 * m * v0**2) -1, 'k-', **plot_settings)


    plt.figure(fig_Etot.fig.number)
    _, _, ymax, ymin = plt.axis()
    #for i_bar in range(int(T_min/T_scale), int(T_max/T_scale)+1):
    #    fig_Etot.plot(
    #        [i_bar, i_bar],
    #        [ymin, ymax],
    #        'k--'
    #    )


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

    ## Save .dat file
    #data.to_csv(filename, sep='\t', index=False)

    if not '--no-show' in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
