import os
import sys
import numpy as np

from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import (
    cg,
    spsolve,
)

import matplotlib.pyplot as plt

#sys.path.append('../qc_dem/')

from .particles import ParticleDistance

def print_matrix(m):
    for i in m:
        s = ''
        for j in i:
            s += '%10.3f' % j
        print(s)

def print_vector(v):
    for i in v:
        print('%10.3f' % i)


class Potential:
    '''
    A base class for calculating forces and stiffness contributions
    due to a potential function.

    Subclasses must implement the following methods:
     - calc_force_ij
     - calc_stiffness_ij
     - calc_conservative_force_ij
    '''
    def __init__(self, particles, alpha=0.5):
        self.__stiffness = None
        self.__last_nlist_update = -1
        self.particles = particles
        self.alpha = alpha

    def calc_force_ij(self, q_i, q_j, i, j, a, d_ij):
        pass
    def calc_conservative_force_ij(self, q_i, q_j, i, j, a, d_ij):
        pass
    def calc_stiffness_ij(self, q_i, q_j, i, j, a, b, d_ij):
        return 0

    def alpha_interp(self, q, qn):
        return (1-self.alpha)*q + self.alpha*qn

    def __resize_stiffness(self, q, neighbor_list, M_over_dt, N):
        dof = 6
        indptr = np.zeros(N+1)#[0]
        indices = []
        stiffness_data = []
        for i, q_i in enumerate(q):
            js = neighbor_list.get_neighbors(i)
            if len(js) == 0:
                # No neighbors for particle i
                indptr[i+1] = len(indices)
                continue
            d_ijs = ParticleDistance.dist_between(q[i, :3], q[js, :3])
            #dr = q[js, :3] - q[i, :3]
            #d_ijs = np.sqrt(
            #    dr[:, 0]**2 +
            #    dr[:, 1]**2 +
            #    dr[:, 2]**2
            #)

            for d_ij, j in zip(d_ijs, js):
                if i==j:
                    #print('Setting diagonal block for i =', i)
                    stiffness_data.append(-M_over_dt[dof*i:dof*(i+1), dof*i:dof*(i+1)])
                else:
                    stiffness_data.append(np.zeros((dof, dof)))
                indices.append(j)
            indptr[i+1] = len(indices)

        stiffness_data = np.array(stiffness_data)
        indices = np.array(indices)

        self.__stiffness = bsr_matrix(
            (stiffness_data, indices, indptr),
            shape=(dof*N, dof*N),
        )

    def calc_conservative_force(self, *args):
        F, _ = self.__calc_force_and_stiffness(
            *args,
            self.calc_conservative_force_ij,
            None, #lambda *a: 0,
        )
        return F

    def calc_force_and_stiffness(self, *args):
        return self.__calc_force_and_stiffness(
            *args,
            self.calc_force_ij,
            self.calc_stiffness_ij,
        )

    def __calc_force_and_stiffness(self, q, d, neighbor_list, M, dt, force_fun, stiffness_fun):
        N = q.shape[0]

        if self.__last_nlist_update < neighbor_list._n_updates:
            # Update the stiffness matrix if the neighbor_list changed
            self.__last_nlist_update =  neighbor_list._n_updates
            self.__resize_stiffness(q, neighbor_list, M/dt, N)

        dof = 6
        energy = 0
        force = np.zeros((dof*N))

        zero_dof = np.zeros((dof, dof))
        ##

        i_interaction = 0
        count = 0
        #for i, q_i in enumerate(q):
        for i in range(N):
            #if type(neighbor_list) == cKDTree:
            #    d_ijs, js = neighbor_list.query(
            #        q_i[:3],
            #        k=7,
            #        distance_upper_bound=1.001*d)
            #    d_ijs = d_ijs[1:]
            #    js = js[1:]
            #else:
            js = neighbor_list.get_neighbors(i)
            if len(js) == 0:
                continue
            d_ijs = ParticleDistance.dist_between(q[i, :3], q[js, :3])
            #dr = q[js, :3] - q[i, :3]
            #d_ijs2 = np.sqrt(
            #    dr[:, 0]**2 +
            #    dr[:, 1]**2 +
            #    dr[:, 2]**2
            #)
            q_i = q[i, :]

            q_js = ParticleDistance.correct_for_bc(q_i, q[js, :])
            #if i == 2:
            #    ic(i, js, q_js[:, :2]) #, d_ijs, d_ijs2)
            # Correct for periodic boundary conditions


            for d_ij, q_j, j in zip(d_ijs, q_js, js):
                if d_ij == float('Inf'):
                    self.__stiffness.data[i_interaction] = zero_dof #np.zeros((dof, dof))

                    #count += 1
                #elif i==j:
                #    # Diagonal blocks won't change after initialization
                #    pass
                elif not i==j:
                    ## Sum over dimensional indices
                    for a in range(dof):
                        #f_update = self.calc_force_ij(q, i, j, a, d_ij)
                        f_update = force_fun(q_i, q_j, i, j, a, d_ij)
                        #if i == 2:
                        #    ic(i, j, a, f_update)
                        force[dof*i+a] += f_update

                        if not stiffness_fun is None:
                            for b in range(dof):
                                #s_update = self.calc_stiffness_ij(q, i, j, a, b, d_ij)
                                s_update = stiffness_fun(q_i, q_j, i, j, a, b, d_ij)
                                self.__stiffness.data[i_interaction][a, b] = -dt * s_update

                i_interaction += 1
        #print('count =', count)
        return (force, self.__stiffness)

class HookianContact(Potential):
    '''
    Calculates inter-particle forces using the Hookian contact model, (normal elastic,
    normal dashpot and tangential dashpot).
    '''
    def __init__(self, particles, dt, alpha=0.5, bonds=None): #k, d, gamma_n, gamma_t, q_n, dt, v):
        Potential.__init__(self, particles, alpha=alpha)
        self.k = particles.k
        self.d = particles.d
        self.m = particles.m
        self.gamma_n = particles.gamma_n
        self.gamma_t = particles.gamma_t
        #self.q_n = q_n
        self.dt = dt
        self.bonds = bonds
        #self.v = v
        #self.particles = particles
        #self.alpha = alpha

    #def calc_force_and_stiffness(self, *args):
    #    self.q_n = self.particles.gen_coords
    #    self.v = self.particles.gen_vel
    #    #print((self.q_n - args[0])/self.dt - self.v)
    #    return Potential.calc_force_and_stiffness(self, *args)


    def __F_d_n(self, v, q_i, q_j, i, j, a, d_ij):
        # Normal dissipation force
        v_dot_n = 0
        for comp in range(3):
            v_dot_n += (1/(d_ij) *
                        (v[i, comp] - v[j, comp]) *
                        (q_i[comp] - q_j[comp]))

        m_eff = self.m[i] * self.m[j] / (self.m[i] + self.m[j])
        res = self.gamma_n * m_eff / d_ij * (q_i[a] - q_j[a]) * v_dot_n
        #if not res == 0:
        #    print('F_d_n =')
        #    print(res)
        #    print(' (d_ij={0}, v_dot_n={1}, q[i, a] - q[j, a]={2})'.format(
        #        d_ij,
        #        v_dot_n,
        #        q[i, a] - q[j, a],
        #    ))
        return res

    def __S_d_n(self, q_i, q_j, i, j, a, b, d_ij):
        # Normal dissipation stress
        if a < 3 and b < 3:
            result = 0.5*self.gamma_n/(d_ij**2 * self.dt) * (q_i[a] - q_j[a]) * (q_i[b] - q_j[b])
            #ic(result)
            return result
        else:
            return 0

    def __F_e_n(self, q_i, q_j, i, j, a, d_ij):
        # Normal elastic force
        
        result = (-self.k * ((self.d[i]/2 + self.d[j]/2)/d_ij - 1) * (q_i[a] - q_j[a]))
        #if type(self) == HookianContact and (a == 0 or a == 1) and abs(result) > 1e-5:
        #    print(i, j, result, type(self))
        return result

    def __F_d_t(self, v, q_i, q_j, i, j, a, d_ij):
        # Tangential dissipation force
        v_dot_n = 0
        for comp in range(3):
            v_dot_n += (1/(d_ij) *
                        (v[i, comp] - v[j, comp]) *
                        (q_i[comp] - q_j[comp]))

        a1 = (a+1) % 3
        a2 = (a+2) % 3
        a1_rot = a1 + 3
        a2_rot = a2 + 3

        m_eff = self.m[i] * self.m[j] / (self.m[i] + self.m[j])
        res = self.gamma_t * m_eff * (
            ## Relative veloctiy
            v[i, a] - v[j, a]
            ## Subtract the normal component
            - 1/(d_ij) * (q_i[a] - q_j[a]) * v_dot_n
            ## Cross product (domega x dr)
            -1/(2) * (
                (v[i, a1_rot] + v[j, a1_rot]) * (q_i[a2] - q_j[a2])
                -
                (v[i, a2_rot] + v[j, a2_rot]) * (q_i[a1] - q_j[a1])
            )
        )

        #if not res == 0:
        #    print('F_d_t =')
        #    print(res)
        return res


    def __T_d_t(self, q_i, q_j, i, j, a, d_ij):
        # Torque due to tangential dissipation force

        # R_ij x F_t
        a1 = (a+1) % 3
        a2 = (a+2) % 3
        a_lin = a - 3

        v = self.particles.gen_vel
        #TODO:
        F_tan_a1 = 1.0 * (
            #self.__F_d_t((self.q_n - q)/self.dt, q, i, j, a1, d_ij)
            + self.__F_d_t(v, q_i, q_j, i, j, a1, d_ij)
        )

        #TODO:
        F_tan_a2 = 1.0 * (
            #self.__F_d_t((self.q_n - q)/self.dt, q, i, j, a2, d_ij)
            + self.__F_d_t(v, q_i, q_j, i, j, a2, d_ij)
        )

        return -0.5*(
            (q_i[a1] - q_j[a1]) * F_tan_a2 -
            (q_i[a2] - q_j[a2]) * F_tan_a1
        )


    def calc_conservative_force_ij(self, q_i, q_j, i, j, a, d_ij):
        #if d_ij > self.particles.d:
        if d_ij > (self.particles.d[i]/2 + self.particles.d[j]/2) or (not self.bonds is None and j in self.bonds.bonds[i]):
            return 0
        return self._calc_conservative_force_ij(q_i, q_j, i, j, a, d_ij)

    def _calc_conservative_force_ij(self, q_i, q_j, i, j, a, d_ij):

        if a >= 3:
            # a in [3, 4, 5] is for torque components
            return 0

        return self.__F_e_n(q_i, q_j, i, j, a, d_ij)

    def calc_force_ij(self, q_i, q_j, i, j, a, d_ij):
        if d_ij > (self.particles.d[i]/2 + self.particles.d[j]/2) or (not self.bonds is None and j in self.bonds.bonds[i]):
            #print('d_{0}{1} > d'.format(i, j))
            return 0

        return self._calc_force_ij(q_i, q_j, i, j, a, d_ij)

    def _calc_force_ij(self, q_i, q_j, i, j, a, d_ij):
        if a >= 3:
            # a in [3, 4, 5] is for torque components
            return self.__T_d_t(q_i, q_j, i, j, a, d_ij)

        F_normal_el = self.__F_e_n(q_i, q_j, i, j, a, d_ij)

        v = self.particles.gen_vel
        F_normal_dis = 1.0 * (
            #self.__F_d_n((self.q_n - q)/self.dt, q, i, j, a, d_ij)
            + self.__F_d_n(v, q_i, q_j, i, j, a, d_ij)
        )
        #TODO:
        #print(
        #    (self.q_n - q)/(self.dt/2)-
        #    self.v
        #    )

        F_tan_dis = 1.0 * (
            #self.__F_d_t((self.q_n - q)/self.dt, q, i, j, a, d_ij)
            + self.__F_d_t(v, q_i, q_j, i, j, a , d_ij)
        )

        return (1-self.alpha)*F_normal_el + F_normal_dis + F_tan_dis

    def calc_stiffness_ij(self, q_i, q_j, i, j, a, b, d_ij):
        return self.__S_d_n(q_i, q_j, i, j, a, b, d_ij)


class BondContact(HookianContact):
    '''
    Calculates the inter particle forces for the Bonded Particle Model described in...
    '''
    def __init__(self, particles, dt):
        #Potential.__init__(self)
        super(BondContact, self).__init__(particles, dt, alpha=0)

        self.particles = particles
        self.dt = dt
        self.bonds = []
        for i in range(particles.N):
            self.bonds.append([])
            for j in range(particles.N):
                if i==j:
                    continue

                d_ij_sq = ((particles.xyz[i, 0] - particles.xyz[j, 0])**2 +
                           (particles.xyz[i, 1] - particles.xyz[j, 1])**2 +
                           (particles.xyz[i, 2] - particles.xyz[j, 2])**2)
                r_cut = 1.01 * (particles.d[i]/2 + particles.d[j]/2)
                if d_ij_sq < r_cut**2:
                    print('Bonded {0}-{1}'.format(i, j))
                    self.bonds[i].append(j)

        #print(self.bonds)

    def calc_force_ij(self, q_i, q_j, i, j, a, d_ij):
        if not j in self.bonds[i]:
            return 0
        force = HookianContact._calc_force_ij(self, q_i, q_j, i, j, a, d_ij)
        ## TODO Check breaking

        return force

    def calc_conservative_force_ij(self, q_i, q_j, i, j, a, d_ij):
        if not j in self.bonds[i]:
            return 0
        force = HookianContact._calc_conservative_force_ij(self, q_i, q_j, i, j, a, d_ij)
        ## TODO Check breaking

        return force

    #def calc_stiffness_ij(self, q, i, j, a, b, d_ij):
    #    return 0

class SumContact(Potential):
    def __init__(self, particles, *args):
        super(SumContact, self).__init__(particles)
        self.contact_models = args

    def calc_force_ij(self, q_i, q_j, i, j, a, d_ij):
        result = sum([
            contact.calc_force_ij(q_i, q_j, i, j, a, d_ij)
            for contact in self.contact_models
        ])
        return result

    def calc_conservative_force_ij(self, q_i, q_j, i, j, a, d_ij):
        result = sum([
            contact.calc_conservative_force_ij(q_i, q_j, i, j, a, d_ij)
            for contact in self.contact_models
        ])
        return result


    def calc_stiffness_ij(self, q_i, q_j, i, j, a, b, d_ij):
        result = sum([
            contact.calc_stiffness_ij(q_i, q_j, i, j, a, b, d_ij)
            for contact in self.contact_models
        ])
        return result

    
def newton_step(dt, q, q0, p0, M_vector, contact_force, contact_stress, dq_alpha=1.0):

    #M = particles.gen_M_matrix/dt
    M_over_dt = M_vector/dt
    q0_ = q0.flatten()
    p0_ = p0.flatten()
    q_ = q.flatten()

    ### Force
    full_force = p0_ - M_over_dt * (q_ - q0_) - dt * contact_force

    e = np.linalg.norm(full_force)

    ## Cholesky decomposition only works for positive definite stress, and should throw
    ## an exception if this is not the case.
    #np.linalg.cholesky(full_stress)

    #if type(contact_stress) == np.ndarray:#np.all(contact_stress == 0):
    #    delta = full_force / -contact_stress.diagonal() #M_over_dt
    #    info = 0
    #else:
        #full_stress = hessian()
    delta, info = cg(contact_stress, -full_force, tol=1e-11)

    if info > 0:
        print('Conjugate gradient method in the Newton step did not converge.')
        p0.shape = (q0.shape[0], 6)
        ic(p0)
        ic(contact_force)
        exit(info)

    ## Reshape the delta
    dof = delta.shape[0]//q0.shape[0]
    delta.shape = (q0.shape[0], dof)

    return (e, delta)

def pf(f):
    print_f = f.copy()
    print_f.shape = (f.size//6, 6)
    return print_f[:, :2]


def newton_solve(particles, dt, q0, ext_forces, e_tol=1e-9, d_tol=1e-13, max_steps=200, walls=[], gravity=None, nlist=None, contact_law=None, dq_alpha=1.0):
    '''
    Find the next position by Newton's method.
    '''
    kd = particles.k * particles.d.mean() * dt
    e = 1/kd
    q0_ = q0.flatten()
    p0 = particles.gen_mom.copy()
    p0_ = p0.flatten()
    M_ = particles.gen_M_vector

    # Create a KDTree if no neighbor list is provided
    if nlist is None:
        pos = q0[:, :3]
        pos.shape = (q0.shape[0], 3)
        kdtree = cKDTree(pos)
    else:
        kdtree = None

    # Constant external forces
    #f_ext = -gravity*particles.m
    #t_ext = np.zeros(particles.xyz.shape)
    #ext = np.concatenate((f_ext, t_ext), axis=1).flatten()

    res = 0
    e0 = kd
    step = 1

    prev_e = -1
    while e/e0 > e_tol and step < max_steps:

        ## Calculate forces with walls
        wall_force = np.zeros(q0_.shape)
        for wall in walls:
            q__ = wall.alpha_interp(q0, particles.gen_coords)
            wall_force += wall.calc_force_and_stiffness(q__, particles.gen_vel, particles.d, particles.m)

        q__ = contact_law.alpha_interp(q0, particles.gen_coords)
        contact_force, contact_stress = contact_law.calc_force_and_stiffness(
            q__,
            particles.d,
            nlist or kdtree,
            particles.gen_M_matrix,
            dt
        )

        ## Improve the estimate of the next position
        (e, delta) = newton_step(
            dt,
            particles.gen_coords,
            q0,
            p0,
            M_,
            contact_force + wall_force + ext_forces,
            contact_stress,
            dq_alpha=dq_alpha,
        )

        particles.xyz += dq_alpha * delta[:, :3]
        particles.rot += dq_alpha * delta[:, 3:]

        d = np.linalg.norm(delta)

        if step == 1:
            d0 = np.linalg.norm(d)
            e0 = np.linalg.norm(e)
            if e0 == 0 or d0 == 0:
                #print('No force or displacement at first Newton iteration.')
                break

        if e/e0 < e_tol or d/d0 < d_tol:
            break

        step += 1
    #ic(step, e/e0, d/d0)
    #print('#steps =', step)
    if step == max_steps:
        ic(step, e/e0, d/d0)
        print('Max steps reached')
    return e


def time_step(particles, dt, walls=[], gravity=np.array([0, 0, 0]), nlist=None, contact_law=None, dt_alpha=1.0, dq_alpha=1, external_gen_force=None):

    q0 = particles.gen_coords.copy()

    m3 = np.vstack((particles.m,)*3).transpose()
    I3 = np.vstack((particles.mom_inertia,)*3).transpose()

    if external_gen_force is None:
        external_gen_force = np.zeros(q0.shape)
    else:
        external_gen_force = - external_gen_force 
    external_gen_force[:, :3] += -gravity*m3
    #t_ext = np.zeros(particles.xyz.shape)
    external_gen_force = external_gen_force.flatten()


    # Initial guess for the next position and rotation
    particles.xyz += dt_alpha * dt*particles.p/m3
    particles.rot += dt_alpha * dt*particles.angular_mom/I3

    # Optimize the position based on potential energy
    E_strain = newton_solve(
        particles,
        dt,
        q0,
        (1-contact_law.alpha) * external_gen_force,
        walls=walls,
        gravity=gravity,
        nlist=nlist,
        contact_law=contact_law,
        dq_alpha=dq_alpha,
    )
    #E_strain = newton_solve2(particles, dt, xyz)

    # Calculate the new momentum

    q = contact_law.alpha_interp(q0, particles.gen_coords)#0.5*(q0 + particles.gen_coords)
    F = (contact_law.alpha)*contact_law.calc_conservative_force(
        q,
        particles.d,
        nlist,
        particles.gen_M_matrix,
        dt,
    )

    for wall in walls:
        #q = wall.alpha_interp(q0, particles.gen_coords)#0.5*(q0 + particles.gen_coords)
        F += wall.calc_conservative_force(q, particles.gen_vel, particles.d)

    F += contact_law.alpha * external_gen_force

    F.shape = q0.shape
    particles.p = (particles.xyz - q0[:, :3]) * m3/dt - F[:, :3]*dt
    particles.angular_mom = (particles.rot - q0[:, 3:]) * I3/dt - F[:, 3:]*dt

    dr = particles.xyz - q0[:, :3]
    return dr

