import numpy as np
from scipy.spatial import cKDTree

class DEMCalc:

    def __init__(self):
        pass

    def step(self, particles, n_list, dt):
        pass

class DEMPureShear(DEMCalc):
    '''
    Deform the particles by pure shear at a given shear rate.
    '''
    def __init__(self, shearrate):
        super(DEMPureShear, self).__init__()

        self.gamma = shearrate

    def M(self, dt):
        return np.array([[1, dt*self.gamma, 0],
                         [dt*self.gamma, 1, 0],
                         [0, 0, 1]])

    def step(self, particles, n_list, dt):
        particles.xyz = np.matmul(particles.xyz, self.M(dt))



class DEMHookianPotential(DEMCalc):
    '''
    Calculate the Hookian potential for a given configuration of particles at
    each time step.
    '''

    def __init__(self, k):
        super(DEMCalc, self).__init__()

        self.k = k

    def step(self, particles, n_list, dt):
        return DEMHookianPotential.eval(
            particles.xyz,
            particles.d,
            self.k,
            n_list=n_list,
        )


    @classmethod
    def eval(cls, positions, d, k, n_list=None):

        if n_list is None:
            n_list = cKDTree(positions)

        V = 0

        for i in range(positions.shape[0]):
            distances, _ = n_list.query(
                positions[i, :],
                k=7,
                distance_upper_bound=1.01*d)

            V += sum([ 0.5 * k * (d - dist)**2
                       for dist in distances[1:]
                       if (d-dist) > 0 and not dist == float('Inf')])
            #for dist in distances[1:]:
            #    overlap = d - dist
            #    if overlap < 0 or overlap == float('Inf'):
            #        break
            #    V += 0.5 * self.k * overlap**2
        print('V = {0:.4}'.format(V))
        return V


class DEM:

    def __init__(self):
        self.particles = None
        self.updates = []


    def step(self, dt):
        if not self.particles:
            # Do nothing if there are no particles
            return

        # Build the nearest neighbour search tree
        neighbours = cKDTree(self.particles.xyz)

        # Compute all the update
        for update in self.updates:
            update.step(self.particles, neighbours, dt)

    def run(self, N, dt):
        for s in range(N):
            self.step(dt)


if __name__ == '__main__':
    print('Hello world')
