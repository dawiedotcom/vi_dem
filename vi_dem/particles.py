import numpy as np


class ParticleDistance:
    '''
    Calculates the distance between particles.
    '''
    #def __init__(self):
    periodic_bc = []

    @classmethod
    def length(cls, vect):
        if len(vect.shape) == 1:
            return np.linalg.norm(vect)
        else:
            return np.linalg.norm(vect, axis=1)

    @classmethod
    def dist_between(cls, pos_i, pos_j):
        dist = cls.length(pos_i - pos_j)
        #ic(pos_i)
        #ic(pos_j)
        for bc in cls.periodic_bc:
            #ic(bc.period)
            test_dist = cls.length(pos_i - pos_j - bc.period)
            #ic(test_dist)
            dist = np.min((dist, test_dist), axis=0)

            test_dist = cls.length(pos_i - pos_j + bc.period)
            dist = np.min((dist, test_dist), axis=0)

        return dist

    @classmethod
    def correct_for_bc(cls, pos_i, pos_js):
        result_pos_js = pos_js.copy()
        for bc in cls.periodic_bc:
            period_length = np.linalg.norm(bc.period)
            idx_far = np.where(cls.length(pos_js[:, :3] - pos_i[:3]) > period_length/2)

            #direction_js = np.dot(pos_js - bc.midpoint, bc.period)
            direction_i = np.dot(pos_i[:3] - bc.midpoint, bc.period)
            #ic(direction_i)
            direction_norm = np.linalg.norm(direction_i)
            if direction_norm == 0:
                direction_i = 1.0
            else:
                direction_i /= direction_norm

            result_pos_js[idx_far, :3] = result_pos_js[idx_far, :3] + direction_i * bc.period

        return result_pos_js

class Particles:
    def __init__(self):
        def empty_array():
            return np.array([], dtype=np.float32)

        #self.x = empty_array()
        #self.y = empty_array()
        #self.z = empty_array()
        self.xyz = empty_array()
        self.p = empty_array()
        self.m = empty_array()

        self.rot = empty_array()
        self.angular_mom = empty_array()

        self.is_rep = np.zeros([], dtype=bool)

        self.d = 0
        self.k = 0
        self.gamma_n = 0
        self.gamma_t = 0

        self.__mass_inertia_matrix = np.zeros([])

    @property
    def v(self):
        return self.p / np.vstack((self.m, ) * 3).transpose()

    @property
    def angular_v(self):
        return self.angular_mom / np.vstack((self.mom_inertia,)*3).transpose()

    @property
    def mom_inertia(self):
        #return 2./5. * self.m * self.d * self.d/4
        return 2./5. * self.m * (self.d/2)**2

    def __concat(self, *args):
        return np.concatenate(args, axis=1)

    @property
    def gen_coords(self):
        return self.__concat(self.xyz, self.rot)

    @property
    def gen_mom(self):
        return self.__concat(self.p, self.angular_mom)

    @property
    def gen_vel(self):
        return self.__concat(self.v, self.angular_v)

    @property
    def gen_M_vector(self):
        #return self.__concat(self.m, self.mom_inertia)
        return self.__mass_inertia_vector

    @property
    def gen_M_matrix(self):
        #return self.__concat(self.m, self.mom_inertia)
        return self.__mass_inertia_matrix

    @property
    def N(self):
        return self.xyz.shape[0]

    def get_rep_position(self):
        return self.xyz[np.where(self.is_rep)[0], :]

    def dist_to(self, x, y, z):
        if self.xyz.shape[0] == 0:
            return np.array([])

        #return np.sqrt((self.xyz[:, 0] - x)**2 +
        #               (self.xyz[:, 1] - y)**2 +
        #               (self.xyz[:, 2] - z)**2)
        test_point = np.array([x, y, z])
        return ParticleDistance.dist_between(test_point, self.xyz)

    def closest_to(self, x, y, z):
        return np.argmin(self.dist_to(x, y, z))

    def mark_rep(self, condition):
        idx = np.where(condition)
        self.is_rep[idx] = True

    def mark_rep_closest_to(self, *args):
        idx = self.closest_to(*args)
        self.is_rep[idx] = True

    def append(self, pos, vel, mass):
        if self.xyz.shape[0] == 0:
            self.xyz = np.zeros((1, 3))
            self.xyz[0] = pos
            self.p = np.zeros((1, 3))
            self.p[0] = vel/mass
            self.rot = np.zeros((1, 3))
            self.angular_mom = np.zeros((1, 3))
            self.m = np.zeros((1, 3))
            self.m[0] = mass*np.ones(3)
        else:
            self.xyz = np.concatenate((self.xyz, (pos,)))
            self.p = np.concatenate((self.p, (vel/mass,)))
            self.rot = np.concatenate((self.rot, np.zeros((1, 3))))
            self.angular_mom = np.concatenate((self.angular_mom, np.zeros((1, 3))))
            self.m = np.concatenate((self.m, (mass*np.ones(3),)))

        self.updateM()

    def updateM(self):
        m = self.m.copy()
        m = np.vstack((m,)*3).transpose()
        I = self.mom_inertia.copy()
        I = np.vstack((I,)*3).transpose()
        self.__mass_inertia_vector = self.__concat(m, I).flatten()
        self.__mass_inertia_matrix = np.diag(self.__mass_inertia_vector)
        #print('mass inertia matrix=')
        #print(self.__mass_inertia_matrix)

    def save(self, fname):
        def reshaped(a):
            result = a.copy()
            result.shape += (1,)
            return result

        arr = np.concatenate((
            self.xyz,
            self.p,
            self.rot,
            self.angular_mom,
            reshaped(self.d),
            reshaped(self.m),
            reshaped(self.is_rep),
        ), axis=1)
        np.save(fname, arr)

    @classmethod
    def load(cls, fname):
        particles = Particles()
        arr = np.load(fname)
        particles.xyz = arr[:, 0:3].copy()
        particles.p = arr[:, 3:6].copy()
        particles.rot = arr[:, 6:9].copy()
        particles.angular_mom = arr[:, 9:12].copy()
        particles.d = arr[:, 12].copy()
        particles.m = arr[:, 13].copy()
        particles.is_rep = arr[:, 14] == 1

        particles.updateM()

        return particles

    @classmethod
    def gen_2d_hex(cls, boundary, particle_diameter=1):
        y_step = np.sqrt(3)/2*particle_diameter
        top_n_rows = int(boundary.yhi/y_step)
        y_start = -top_n_rows * y_step
        y_end = top_n_rows * y_step
        num_rows = 2*top_n_rows + 1

        x_start1 = int(boundary.xlo/particle_diameter) * particle_diameter
        x_end1 = int(boundary.xhi/particle_diameter) * particle_diameter
        x_start2 = (0.5 + int(boundary.xlo/particle_diameter)) * particle_diameter
        x_end2 = (0.5 + int(boundary.xhi/particle_diameter)) * particle_diameter

        xyz = None

        ic(x_start1, x_end1, x_start2, x_end2)

        n_added = 0
        #for i, y_coord in enumerate(np.arange(y_start, y_end, y_step)):
        for i in np.arange(-top_n_rows, top_n_rows + 1):
            y_coord = y_step * i
            if i%2==0:
                x_new =np.arange(x_start1, x_end1+particle_diameter, particle_diameter)
            else:
                x_new =np.arange(x_start2, x_end2, particle_diameter)

            n_new = x_new.size
            if xyz is None:
                xyz = np.zeros((n_new, 3))
            else:
                xyz = np.append(xyz, np.zeros((n_new, 3)), axis=0)
            xyz[n_added:(n_added+n_new), 0] = x_new
            xyz[n_added:(n_added+n_new), 1] = y_coord * np.ones(n_new)
            n_added += n_new

        #print(list(zip(x, y)))

        particles = Particles()
        particles.xyz = xyz
        particles.p = np.zeros(xyz.shape)
        particles.m = np.ones(xyz.shape[0])
        particles.d = particle_diameter * np.ones(particles.xyz.shape[0])
        particles.is_rep = np.zeros(xyz.shape[0], dtype=bool)

        particles.rot = np.zeros(xyz.shape)
        particles.angular_mom = np.zeros(xyz.shape)

        return particles



class NList():
    '''
    Implements a neighbor list using the velocity Verlet method.
    '''
    def __init__(self, r_skin):
        self._r_skin = r_skin
        self._n_list = []
        self._dr_since_update = np.zeros((0,3))
        self._n_updates = 0

    def __del__(self):
        print('Neighbor list updates: {0}'.format(self._n_updates))

    def get_neighbors(self, i):
        return self._n_list[i]

    def time_step(self, particles, dr):

        N = dr.shape[0]

        if not N == particles.N or not N == self._dr_since_update.shape[0]:
            self.update(particles)

        self._dr_since_update += dr

        #drneimax, drneimax2 = 0, 0

        #dr_norm = np.sqrt(
        #    self._dr_since_update[:, 0]**2 +
        #    self._dr_since_update[:, 1]**2 +
        #    self._dr_since_update[:, 2]**2
        #)
        dr_norm = ParticleDistance.length(self._dr_since_update)

        dr_2max = -np.partition(-dr_norm, 1)[:2] if dr_norm.size >= 2 else dr_norm

        if np.sum(dr_2max) > self._r_skin:
            self.update(particles)


    def update(self, particles):
        print('Update nlist')
        self._n_updates += 1
        #if not len(self._n_list) == particles.N:
        #    self._resize()

        self._dr_since_update = np.zeros(particles.xyz.shape)

        N = particles.N
        # Create a list containing N empty lists
        self._n_list = []

        r_cut = self._r_skin + np.min(particles.d)
        for i in range(N):
            d_ijs = ParticleDistance.dist_between(particles.xyz[i, :], particles.xyz)
            js = np.where(d_ijs < r_cut)[0]
            self._n_list.append(js)
            #self._n_list.append([])
            #for j in range(N):
            #    if i == j:
            #        self._n_list[i].append(j)
            #        continue

            #    d_ij_sq = ((particles.xyz[i, 0] - particles.xyz[j, 0])**2 +
            #               (particles.xyz[i, 1] - particles.xyz[j, 1])**2 +
            #               (particles.xyz[i, 2] - particles.xyz[j, 2])**2)
            #    if d_ij_sq < r_cut**2:
            #        self._n_list[i].append(j)

        #print('NList:',self._n_list)

    #def __resize(self):
    #    pass
