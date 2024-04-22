import numpy as np
import copy
import matplotlib.pyplot as plt
from lte import LTE as lte
from ja import JA as ja
from dmp import DMP as dmp
import sys
sys.path.insert(1, './Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR
from TLFSD import TLFSD as tlfsd
from scipy.optimize import minimize
from douglas_peucker import DouglasPeuckerPoints

class _LFD:
    def __init__(self, demos=None, constraints=[], indices=[]):
        self.demos = None
        self.n_pts = None
        self.n_dims = None
        if demos is not None:
            self.demos = np.array(demos)
            (self.n_pts, self.n_dims) = self.demos[0].shape
        self.constraints = np.array(constraints)
        self.indices = np.array(indices)
        self.reproduction = None
        self._generate()

    def get_demo(self):
        return self.demos

    def set_demo(self, new_demos):
        self.demos = np.array(new_demos)
        (self.n_pts, self.n_dims) = self.demos[0].shape
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

    def plot(self, filepath=None, ax=None):
        if ax is None:
            ax = plt.gca

        if self.reproduction is not None:
            if self.n_dims == 2:
                ax.plot(self.demos[0, :, 0], self.demos[0,:, 1], color='black', label='original')
                ax.plot(self.reproduction[:,0], self.reproduction[:,1], color='blue', label='reproduction')
                ax.scatter(self.constraints[:,0], self.constraints[:,1], marker='x')
            elif self.n_dims == 3:
                ax.plot(self.demos[0, :, 0], self.demos[0, :, 1], self.demos[0,:,2], color='black', label='original')
                ax.plot(self.reproduction[:, 0], self.reproduction[:, 1], self.demos[:,2], color='blue', label='reproduction')
                ax.scatter(self.constraints[:,0], self.constraints[:,1], self.constraints[:,2], marker='x')

        else:
            print("NO REPRODUCTION IS AVAILABLE")

    def _generate(self):
        raise NotImplementedError

class LTE(_LFD):
    def _generate(self):
        if self.demos is not None:
            self.reproduction=lte(self.demos[0], C=self.constraints, inds=self.indices)

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
        if self.demos is not None:
            self.reproduction = dmp(self.demos[0], self.constraints, self.indices,
                                self.duration, self.dt, self.use_improved, self.k, self.D)

class JA(_LFD):
    def __init__(self, demos=None, constraints=[], indices=[], lmbda = None,time = None, C_vel = None, C_accel = None, method = 'fast'):
        self.lmbda = lmbda
        self.time = time
        self.C_vel = C_vel
        self.C_accel = C_accel
        self.method = method

        new_shape = np.array(constraints).shape + tuple([1])
        super().__init__(demos, np.array(constraints).reshape(new_shape),indices)

    def set_constraints(self, new_constraints, new_indices):
        new_shape = np.array(new_constraints).shape + tuple([1])
        super().set_constraints(np.array(new_constraints).reshape(new_shape), new_indices)

    def _generate(self):
        if self.demos is not None:
            self.reproduction = ja(self.demos[0], self.constraints, self.indices, self.lmbda,
                                self.time, self.C_vel, self.C_accel, self.method)


class GMM(_LFD):
    def __init__(self, demos=None, constraints=None, indices=[], num_states=4):
        #self.other_coords=False
        self.num_states= num_states
        self.demos = None
        self.n_pts = None
        self.n_dims = None
        if demos is not None:
            self.demos = np.array(demos)
            (self.n_pts, self.n_dims) = self.demos[0].shape
        # self.constraints = constraints
        # self.indices = indices
        self._generate()

    def set_demo(self, new_demos):
        self.demos = np.array(new_demos)
        (self.n_pts, self.n_dims) = self.demos[0].shape
        self._generate()

    def _generate(self):
        if self.demos is not None:
            demos = []
            self.t = np.linspace(0, 1, self.n_pts).reshape((self.n_pts, 1))
            for demo in self.demos:
                demos.append(np.hstack((self.t, demo)))
            demos = np.array(demos)
            demos = np.transpose(np.vstack((demos)))
            self.gmm = GMM_GMR(self.num_states)
            self.gmm.fit(demos)
            self.gmm.predict(np.linspace(min(self.t), max(self.t), 100).reshape((100)))
            self.reproduction = np.transpose(self.gmm.getPredictedMatrix())

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca
        if self.reproduction is not None:
            if self.n_dims == 2:
                ax.plot(self.demos[0, :, 0], self.demos[0,:, 1], color='black', label='original')
                ax.plot(self.reproduction[:,1], self.reproduction[:,2], color='blue', label='reproduction')
            elif self.n_dims == 3:
                ax.plot(self.demos[0, :, 0], self.demos[0, :, 1], self.demos[0,:,2], color='black', label='original')
                ax.plot(self.reproduction[:,1], self.reproduction[:,2], self.demos[:,3], color='blue', label='reproduction')

        else:
            print("NO REPRODUCTION IS AVAILABLE")



class TLFSD(_LFD):
    def __init__(self, success=[], failure=[], constraints=[], indices=[],k=None, num_states=4, include_other_system=False):
        self.s_demos=[]
        self.f_demos=[]
        self.n_pts = None
        self.n_dims = None
        if success:
            self.s_demos = self._transform(success)
            (_, self.n_dims) = np.array(self.s_demos[0]).shape
        if failure:
            self.f_demos= self._transform(failure)
            (_, self.n_dims) = np.array(self.s_demos[0]).shape


        self.constraints = np.transpose(constraints)
        self.indices = indices
        self.k = k
        self.num_states = num_states
        self.include_other_system = include_other_system
        self.num_pts = 100
        self.reproduction = None
        self._generate()

    def set_demo(self, new_success=[], new_failure=[]):
        self.s_demos = []
        self.f_demos = []
        if len(new_success) > 0:
            self.s_demos = self._transform(new_success)
            (_, self.n_dims) = np.array(self.s_demos[0]).shape

        if len(new_failure) > 0:
            self.f_demos = self._transform(new_failure)
            (_, self.n_dims) = np.array(self.f_demos[0]).shape
        self._generate()

    def set_constraints(self, new_constraints, new_indices):
        self.constraints = np.transpose(new_constraints)
        self.indices = new_indices
        self._generate()

    def _transform(self, demos):
        new_demos = []
        for demo in demos:
            demo = np.array(demo)
            demo = demo - demo[-1, :]
            demos = demos * 100
            new_demo = DouglasPeuckerPoints(demo, self.num_pts)
            new_demos.append(np.transpose(new_demo))
        return new_demos


    def _generate(self):
        if (len(self.s_demos) > 0 or len(self.f_demos) > 0) and len(self.constraints) > 0:
            model = tlfsd(copy.copy(self.s_demos), copy.copy(self.f_demos))
            model.encode_GMMs(self.num_states, self.include_other_system)
            self.reproduction = np.transpose(model.get_successful_reproduction(self.k, self.indices, self.constraints))
            #print(self.reproduction)
