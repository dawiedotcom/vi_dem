import numpy as np

from .particles import (
    ParticleDistance,
    NList,
)

class WallNList(NList):
    def __init__(self, r_skin, walls):
        super(WallNList, self).__init__(r_skin)

        self.__walls = walls
        for i, wall in enumerate(walls):
            wall.n_list = self #wall_nlist
            wall.n_list_pos = i

    def __del__(self):
        print('Neighbor list updates (walls): {0}'.format(self._n_updates))

    def time_step(self, particles, dr):
        if not len(self.__walls) == len(self._n_list): # or self._n_updates == 0:
            self.update(particles)

        super(WallNList, self).time_step(particles, dr)

    def update(self, particles):
        print('Update wall nlist')
        self._n_updates += 1

        self._dr_since_update = np.zeros(particles.xyz.shape)

        N_walls = len(self.__walls)
        # Create a list containing N empty lists
        self._n_list = [[]] * N_walls

        r_cut = self._r_skin + particles.d/2
        for i, wall in enumerate(self.__walls):
            dist = wall.dist_to(particles.xyz)
            #ic(i, dist)
            self._n_list[i] = np.where(np.abs(dist) < r_cut)[0]

        #print('Wall NList update: ')
        #print(self._n_list)
        #for i,j in zip(range(N_walls), range(N_part)):

        #    d_ij_sq = ((particles.xyz[i, 0] - particles.xyz[j, 0])**2 +
        #               (particles.xyz[i, 1] - particles.xyz[j, 1])**2 +
        #               (particles.xyz[i, 2] - particles.xyz[j, 2])**2)
        #    if d_ij_sq < self.__r_cutt**2:
        #        self.__n_list[i].append(j)

class Plane:
    def __init__(self, position, normal):
        self.position = position
        self.norm = normal / np.linalg.norm(normal)
        
        self.n_list = None
        self.n_list_pos = 0

    def dist_to(self, pos):
        return -(
            self.norm[0] * (self.position[0] - pos[:, 0]) +
            self.norm[1] * (self.position[1] - pos[:, 1]) +
            self.norm[2] * (self.position[2] - pos[:, 2])
        )

    def reflect(self, pos, dist_to=None):
        if dist_to is None:
            dist_to = self.dist_to(pos)
        return pos - 2*np.outer(dist_to, self.norm)


class Wall(Plane):
    '''
    Calculates the elastic component of particle-wall interactions.
    '''
    def __init__(self, position, normal, k, gamma_n, gamma_t, alpha=0.5):
        #self.position = position
        #self.norm = normal / np.linalg.norm(normal)
        super().__init__(position, normal)

        self.k = k
        self.gamma_n = gamma_n
        self.gamma_t = gamma_t

        #self.normal_interactions =
        self.alpha = alpha

    def alpha_interp(self, q, qn):
        return (1-self.alpha)*q + self.alpha*qn

    def calc_force_and_stiffness(self, pos, vel, d, m):
        return self.__calc_force_and_stiffness(pos, vel, d, m, self.calc_force_ij)

    def calc_conservative_force(self, pos, vel, d):
        return self.__calc_force_and_stiffness(pos, vel, d, np.zeros(d.shape), self.calc_conservative_force_ij)

    def __calc_force_and_stiffness(self, pos, vel, d, m, force_func):
        dof = 6
        N = pos.shape[0]
        force = np.zeros((dof*N))
        stiffness = np.zeros((dof*N, dof*N))

        if self.n_list is None:
            dist = self.dist_to(pos)
        else:
            neigh_idx = self.n_list.get_neighbors(self.n_list_pos)
            if neigh_idx.size == 0:
                return force
            dist = self.dist_to(pos[neigh_idx, :])

        idx = np.where(np.abs(dist) < np.max(d)/2)[0]

        energy = 0

        #ic(self.position, neigh_idx, dist)
        for i_, d_ij in zip(idx, dist[idx]):
            i = neigh_idx[i_]

            for a in range(dof):
                force[dof*i + a] += force_func(vel[i], d_ij, d[i], m[i], a)
                #force[dof*i + a] += self.calc_force_ij(vel[i], d_ij, d, a)
                #ic(force)

        return force

    def __dashpot_t(self, v_i, d_ij, m_i, a):
        a1 = (a+1) % 3
        a2 = (a+2) % 3
        a1_rot = a1 + 3
        a2_rot = a2 + 3

        #v_dot_n = sum([v*n for (v, n) in zip(v_i[:3], self.norm)])
        v_dot_n = (v_i[0] * self.norm[0] +
                   v_i[1] * self.norm[1] +
                   v_i[2] * self.norm[2])

        t_dashpot = (v_i[a] - v_dot_n * self.norm[a]  -
                     (v_i[a1_rot] * d_ij * self.norm[a2] -
                      v_i[a2_rot] * d_ij * self.norm[a1]))
        return self.gamma_t * m_i/2 * t_dashpot

    def calc_conservative_force_ij(self, v_i, d_ij, d_i, m_i, a):
        if a < 3 and d_ij < d_i/2:
            spring = -self.k * (d_i/2 - d_ij) * self.norm[a]
            return self.alpha * spring
        return 0

    def calc_force_ij(self, v_i, d_ij, d_i, m_i, a):

        if d_ij > d_i/2:
            return 0

        if a < 3:
            # Normal elastic part
            spring = -self.k * (d_i/2 - d_ij) * self.norm[a]

            # Normal dissipation
            n_dashpot = self.gamma_n * m_i/2 * v_i[a] * self.norm[a]**2

            # Tangential Dissipation
            t_dashpot = self.__dashpot_t(v_i, d_ij, m_i, a)

            return ((1-self.alpha)*spring + n_dashpot + t_dashpot)

        elif a >= 3:
            # Torque
            a1 = (a+1) % 3
            a2 = (a+2) % 3
            a1_rot = a1 + 3
            a2_rot = a2 + 3
            return - (
                d_ij * self.norm[a1] * self.__dashpot_t(v_i, d_ij, d_i, a2)
                -
                d_ij * self.norm[a2] * self.__dashpot_t(v_i, d_ij, d_i, a1)
            )


    def calc_stiffness_ij(self, i, j, a, b, d_ij):
        return 0

class PeriodicWallPair:
    #def __init__
    def __init__(self, position, normal, particles, contact_law): #, alpha=0.5):
        # Position and normal -- of one wall. The second wall's position
        # and normal will be -position and -normal, respectively.

        #self.w1 = Wall( position,  normal, particles.k, particles.gamma_n, particles.gamma_t, alpha=0.5)
        #self.w2 = Wall(-position, -normal, particles.k, particles.gamma_n, particles.gamma_t, alpha=0.5)
        self.w1 = Plane( position,  normal)
        self.w2 = Plane(-position, -normal)

        ParticleDistance.periodic_bc.append(self)

    @property
    def period(self):
        return self.w1.position - self.w2.position
    @property
    def midpoint(self):
        return 0.5*(self.w1.position + self.w2.position)
        #self.n_list = None
        ##self.n_list = WallNList(d/2, [self.w1, self.w2])
        #self.period = np.zeros(6)
        #self.period[:3] = 2*position

        #self.particles = particles
        #self.contact_law = contact_law

        #self.alpha = 0.5

        #self.xyz = np.array([])
        #self.vel = np.array([])
        #self.d = np.array([])
        #self.m = np.array([])

    #def __setattr__(self, name, value):

    #    super().__setattr__(name, value)
    #    if name == 'n_list' or name == 'n_list_pos':
    #        setattr(self.w1, name, value)
    #        setattr(self.w2, name, value)
    #    
    #def calc_dr(self, q):
    #    pass

    """
    def alpha_interp(self, q, qn):
        return (1-self.alpha)*q + self.alpha*qn

    def dist_to(self, pos):
        d1 = self.w1.dist_to(pos)
        d2 = self.w2.dist_to(pos)
        #ic(d1, d2)
        return np.min(np.stack((d1, d2)), axis=0)

    def __update_buffers(self, pos, vel, d, m):

        neigh_idx = self.n_list.get_neighbors(self.n_list_pos)
        N = neigh_idx.size
        if not self.xyz.shape[0] == 2*N:
            self.xyz = np.zeros((2*N, 6))
            self.vel = np.zeros((2*N, 6))
            self.d = np.zeros((2*N))
            self.m = np.zeros((2*N))

        self.xyz[:N, :] = pos[neigh_idx, :]
        self.xyz[N:, :] = pos[neigh_idx, :] + self.period

        self.vel[:N, :] = vel[neigh_idx, :]
        self.vel[N:, :] = vel[neigh_idx, :] 

        self.d[:N] = d[neigh_idx]
        self.d[N:] = d[neigh_idx] 

        self.m[:N] = m[neigh_idx]
        self.m[N:] = m[neigh_idx] 

    def calc_force_and_stiffness(self, pos, vel, d, m):

        #return self.__calc_force_and_stiffness(pos, vel, d, m, self.calc_force_ij)
        self.__update_buffers(pos, vel, d, m)
        result, _ = self.contact_law.calc_force_and_stiffness(self.xyz, self.vel, self.d, self.m)
        #result = (
        #    self.w1.calc_force_and_stiffness(pos, vel, d, m) + 
        #    self.w2.calc_force_and_stiffness(pos, vel, d, m)
        #)
        #ic(result)
        return result

    def calc_conservative_force(self, pos, vel, d):
        #return self.__calc_force_and_stiffness(pos, vel, d, np.zeros(d.shape), self.calc_conservative_force_ij)
        #return (
        #    self.w1.calc_conservative_force(pos, vel, d) + 
        #    self.w2.calc_conservative_force(pos, vel, d)
        #)
        #result = (
        #    self.w1.calc_force_and_stiffness(pos, vel, d, m) + 
        #    self.w2.calc_force_and_stiffness(pos, vel, d, m)
        #)
        #ic(result)
        self.__update_buffers(pos, vel, d, m)
        result = self.contact_law.calc_conservative_force(self.xyz, self.vel, self.d, self.m)
        return result


    def calc_conservative_force_ij(self, v_i, d_ij, d_i, m_i, a):
        result = self.contact_law.calc_conservative_force_ij(v_i, d_ij, d_i, m_i, a)
        return result
        #return (
        #    self.w1.calc_conservative_force_ij(v_i, d_ij, d_i, m_i, a) + 
        #    self.w2.calc_conservative_force_ij(v_i, d_ij, d_i, m_i, a)
        #)

    def calc_force_ij(self, v_i, d_ij, d_i, m_i, a):
        #return (
        #    self.w1.calc_force_ij(v_i, d_ij, d_i, m_i, a) + 
        #    self.w2.calc_force_ij(v_i, d_ij, d_i, m_i, a)
        #)
        result = self.contact_law.calc_force_ij(v_i, d_ij, d_i, m_i, a)
        return result
    """
