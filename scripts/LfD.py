import numpy as np
import matplotlib.pyplot as plt
from lte import LTE as lte
from ja import JA as ja
from dmp import DMP as dmp
import sys
sys.path.insert(1, './Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR

from scipy.optimize import minimize


class _LFD:
    def __init__(self, demos=None, constraints=[], indices=[]):
        self.demo = None
        if demos is not None:
            self.demo = np.array(demos[0])
        self.constraints = np.array(constraints)
        self.indices = np.array(indices)
        self.reproduction = None
        self._generate()

    def get_demo(self):
        return self.demos

    def set_demo(self, new_demos):
        self.demo = np.array(new_demos[0])
        self._generate()

    def get_constraints(self):
        return self.constraints

    def set_constraints(self, new_constraints, new_indices):
        self.constraints = np.array(new_constraints)
        self.indices = np.array(new_indices)
        self._generate()

    def get_reproduction(self):
        if self.reproduction is not None:
            return self.reproduction
        else:
            print("NO REPRODUCTION IS AVAILABLE")
            return None

    def plot(self):
        if self.reproduction is not None:
            plt.plot(self.reproduction[:0], self.reproduction[:1])
            plt.show()
        else:
            print("NO REPRODUCTION IS AVAILABLE")



    def _generate(self):
        raise NotImplementedError

class LTE(_LFD):
    def _generate(self):
        if self.demo is not None:
            self.reproduction=lte(self.demo, C=self.constraints, inds=self.indices)

class DMP(_LFD):
    def __init__(self, demos=None, constraints=[], indices=[],
                 duration=1.0, dt = None, use_improved=True, k=None,D=None):
        self.duration = duration
        self.dt = dt
        self.use_improved = use_improved
        self.k = k
        self.D = D
        super().__init__(demos,constraints,indices)


    def _generate(self):
        if self.demo is not None:
            self.reproduction = dmp(self.demo, self.constraints, self.indices,
                                self.duration, self.dt, self.use_improved, self.k, self.D)

class JA(_LFD):
    def __init__(self, demo=None, constraints=[], indices=[], lmbda = None,time = None, C_vel = None, C_accel = None, method = 'fast'):
        self.lmbda = lmbda
        self.time = time
        self.C_vel = C_vel
        self.C_accel = C_accel
        self.method = method

        new_shape = np.array(constraints).shape + tuple([1])
        super().__init__(demo, np.array(constraints).reshape(new_shape),indices)

    def set_constraints(self, new_constraints, new_indices):
        new_shape = np.array(new_constraints).shape + tuple([1])
        super().set_constraints(np.array(new_constraints).reshape(new_shape), new_indices)

    def _generate(self):
        if self.demo is not None:
            self.reproduction = ja(self.demo, self.constraints, self.indices, self.lmbda,
                                self.time, self.C_vel, self.C_accel, self.method)


class GMM(_LFD):
    def __init__(self, demos=None, constraints=None, indices=[], num_states=4):
        #self.other_coords=False
        self.num_states= num_states
        self.demo = None
        #super().__init__(demos, constraints, indices)
        if demos is not None:
            self.demo = np.transpose(np.array(demos[0]))
        self.constraints = constraints
        self.indices = indices

    def LTE_ND_any_constraints(self,org_traj, constraints, index):
        fixedWeight = 1e9
        (nbNodes, nbDims) = np.shape(org_traj)
        L = 2. * np.diag(np.ones((nbNodes,))) - np.diag(np.ones((nbNodes - 1,)), 1) - np.diag(np.ones((nbNodes - 1,)),
                                                                                              -1)
        L[0, 1] = -2.
        L[-1, -2] = -2.
        L = L / 2.
        # not how it works in above code
        delta = np.matmul(L, org_traj)
        to_append_L = np.zeros((len(index), nbNodes))
        for i in range(len(index)):
            to_append_L[[i], index[i]] = fixedWeight
            to_append_delta = fixedWeight * np.array(constraints[i])
            delta = np.vstack((delta, to_append_delta))

        L = np.vstack((L, to_append_L))
        new_traj, _, _, _ = np.linalg.lstsq(L, delta, rcond=-1)
        # new_traj, _, _, _ = np.linalg.solve(L, delta)
        # print(np.shape(new_traj))
        return new_traj

    def get_constraint_cost(self, X):
        self.traj = X.reshape((self.n_dims, self.n_pts))
        sum = 0.
        for i in range(len(self.indices)):
            sum += np.linalg.norm(self.traj[:, self.indices[i]] - self.constraints[i]) ** 2
        return 1e12 * sum

    def set_demo(self, new_demos):
        self.demo = np.transpose(np.array(new_demos[0]))
        (self.n_dims, self.n_pts) = self.demo.shape
        self.t = np.linspace(0, 1, self.n_pts).reshape((1, self.n_pts))
        self.demo = np.vstack((self.t, self.demo))
        self.s_gmm = GMM_GMR(self.num_states)
        self.s_gmm.fit(np.hstack([self.demo]))
        self.s_gmm.predict(self.t)
        self.mu_s = self.s_gmm.getPredictedData()
        self.cov_s = self.s_gmm.getPredictedSigma()
        self.inv_cov_s = np.zeros((np.shape(self.cov_s)))
        for i in range(self.n_pts):
            self.inv_cov_s[:, :, i] = np.linalg.inv(self.cov_s[:, :, i])
        self._generate()

    def _generate(self):
        if self.demo is not None:
            suggest_traj = self.mu_s[1:,:]
            #print("mu_s",self.mu_s[1:,:].shape)
            #self.reproduction = suggest_traj
            suggest_traj = np.transpose(self.LTE_ND_any_constraints(
                np.transpose(suggest_traj), self.constraints, self.indices))

            #self.reproduction= obj.get_successful_reproduction(constraints=self.constraints, indices=self.indices)
            res = minimize(self.get_constraint_cost, suggest_traj.flatten(), tol=1e-6)

            self.reproduction = np.transpose(np.reshape(res.x,(self.n_dims, self.n_pts)))