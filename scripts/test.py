import numpy as np
import h5py
import matplotlib.pyplot as plt
from LfD import *
from lte import LTE as lte
import sys
sys.path.insert(1, './Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR
import matplotlib.patches as mpatches
from scipy.optimize import minimize
#from movement_primitives.promp import ProMP

def get_lasa_trajN(shape_name, n=1):
    # ask user for the file which the playback is for
    # filename = raw_input('Enter the filename of the .h5 demo: ')
    # open the file
    filename = 'lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    # navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo' + str(n))
    pos_info = demo.get('pos')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    # close out file
    hf.close()
    x = np.reshape(x_data, ((len(x_data), 1)))
    y = np.reshape(y_data, ((len(y_data), 1)))
    return np.hstack((x, y))


def test(model_name,filepath, lmbda=None):
    lasa_names = ['Angle', 'BendedLine', 'CShape', 'DoubleBendedLine', 'GShape', \
                  'heee', 'JShape', 'JShape_2', 'Khamesh', 'Leaf_1', \
                  'Leaf_2', 'Line', 'LShape', 'NShape', 'PShape', \
                  'RShape', 'Saeghe', 'Sharpc', 'Sine', 'Snake', \
                  'Spoon', 'Sshape', 'Trapezoid', 'Worm', 'WShape', \
                  'Zshape']

    fig, axs = plt.subplots(ncols=7, nrows=4)
    fig_x = 0
    fig_y = 0
    colors = {
        "original": "black",
        "new_traj1": "green",
        "new_traj2": "blue",
        "new_traj3": "orange",
    }

    original_patch = mpatches.Patch(color='black', label='original')
    new_traj1_patch = mpatches.Patch(color='green', label='new_traj1')
    new_traj2_patch = mpatches.Patch(color='blue', label='new_traj2')
    new_traj3_patch = mpatches.Patch(color='orange', label='new_traj3')

    if model_name == "LTE":
        model=LTE()
    elif model_name == "DMP":
        model=DMP()
    elif model_name == "JA":
        model=JA()
    else:
        model = GMM()
    for name in lasa_names:
        traj = get_lasa_trajN(name)
        #print("traj", traj.shape)
        x_demo = traj[:, 0]
        y_demo = traj[:, 1]
        num_points = 1000

        new_traj = []
        model.set_demo([traj])


        model.set_constraints([[x_demo[0] + 10, y_demo[0] + 10],
                             np.array([x_demo[-1] + 10, y_demo[-1] + 10])],
                            [0, num_points - 1])

        new_traj.append(model.get_reproduction())

        model.set_constraints([np.array([x_demo[0] - 10, y_demo[0] - 10]),
                               np.array([x_demo[-1] - 10, y_demo[-1] - 10])],
                              [0, num_points - 1])

        new_traj.append(model.get_reproduction())

        model.set_constraints([np.array([x_demo[0] + 20, y_demo[0] + 20]),
                               np.array([x_demo[20] + 20, y_demo[20] + 20]),
                               np.array([x_demo[-1] + 20, y_demo[-1] + 20])],
                              [0, 20, num_points - 1])

        new_traj.append(model.get_reproduction())



        # if model != "DMP":
        # new_traj.append(get_reproduction(model, demo=traj,\
        #                                  C=[np.array([x_demo[0] + 10, y_demo[0] + 10]),
        #                                     np.array([x_demo[-1] + 10, y_demo[-1] + 10])], \
        #                                  inds=[0, num_points - 1], lmbda=lmbda))
        # new_traj.append(get_reproduction(model, demo=traj, \
        #                                  C=[np.array([x_demo[0] - 10, y_demo[0] - 10]),
        #                                     np.array([x_demo[-1] - 10, y_demo[-1] - 10])], \
        #                                  inds=[0, num_points - 1], lmbda=lmbda))
        # new_traj.append(get_reproduction(model, demo=traj, \
        #                                  C=[np.array([x_demo[0] + 20, y_demo[0] + 20]),
        #                                     np.array([x_demo[20] + 20, y_demo[20] + 20]),
        #                                     np.array([x_demo[-1] + 20, y_demo[-1] + 20])], \
        #                                  inds=[0, 20, num_points - 1], lmbda=lmbda, k=0.1, D=10))

        axs[fig_x, fig_y].plot(traj[:, 0], traj[:, 1], color='black', label='original')
        a = 1
        for each in new_traj:
            axs[fig_x, fig_y].plot(each[:,0], each[:,1], color=colors['new_traj' + str(a)])
            a += 1

        fig_y += 1
        if fig_y >= 7:
            fig_y = 0
            fig_x += 1

    fig.legend(handles=[original_patch, new_traj1_patch, new_traj2_patch, new_traj3_patch])
    fig.suptitle(model_name)
    plt.savefig(filepath)
    # plt.show()

if __name__ == '__main__':
    test("LTE", "test_results/LTE.png")
    test("DMP", "test_result/DMP.png")
    test("JA", "test_result/JA.png", 10000)
    test("GMM","test_result/GMM.png")


