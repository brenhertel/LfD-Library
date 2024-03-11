import numpy as np
import h5py
import matplotlib.pyplot as plt
from ja import JA
from dmp import DMP
from lte import LTE
from TLFSD import TLFSD
import matplotlib.patches as mpatches


def get_reproduction(model="LTE", **kwargs):
    if model == "LTE":
        demo = []
        C = None
        inds = []
        for key, value in kwargs.items():
            if key == "demo":
                demo = value
            elif key == "C":
                C = value
            elif key == "inds":
                inds = value
        return LTE(demo, C, inds)
    elif model == "DMP":
        demo = []
        C = None
        inds = []
        duration = 1.0
        dt = None
        use_improved = True
        k = None
        D = None
        for key, value in kwargs.items():
            if key == "demo":
                demo = value
            elif key == "C":
                C = value
            elif key == "inds":
                inds = value
            elif key == "duration":
                duration = value
            elif key == "dt":
                dt = value
            elif key == "use_improved":
                use_improved = value
            elif key == 'k':
                k = value
            elif key == "D":
                D = value
        return DMP(demo, C, inds, duration, dt, use_improved, k, D)
    elif model == "JA":
        demo = []
        C = None
        inds = []
        lmbda = None
        time = None
        C_vel = None
        C_accel = None
        method = 'fast'
        for key, value in kwargs.items():
            if key == "demo":
                demo = value
            elif key == "C":
                C = value
            elif key == "inds":
                inds = value
            elif key == "lmbda":
                lmbda = value
            elif key == "time":
                time = value
            elif key == "C_vel":
                C_vel = value
            elif C_accel == "C_accel":
                C_accel = value
            elif key == "method":
                method = value
        return JA(demo, C, inds, lmbda, time, C_vel, C_accel, method)

    # NEED UPDATE#
    # elif model == "GMM":
    #     successes = []
    #     failures = []
    #     indices = []
    #     constraints = []
    #     num_states = 4
    #     include_other_systems = False
    #     k = None
    #     for key, value in kwargs.items():
    #         if key == "successes":
    #             successes = value
    #         elif key == "failures":
    #             failures = value
    #         elif key == "indices":
    #             indices = value
    #         elif key == "constraints":
    #             constraints = "constraint"
    #         elif key == "num_states":
    #             num_states = value
    #         elif key == 'include_other_systems':
    #             include_other_systems = value
    #         elif key == "K":
    #             k = value
    #     obj = TLFSD(successes, failures)
    #     obj.encode_GMMs(num_states, include_other_systems)
    #     return obj.get_successful_reproduction(k, indices, constraints)


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


def test(model, filepath, lmbda=None):
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

    for name in lasa_names:
        traj = get_lasa_trajN(name)
        x_demo = traj[:, 0]
        y_demo = traj[:, 1]
        num_points = 1000
        x_demo = x_demo.reshape((num_points, 1))
        y_demo = y_demo.reshape((num_points, 1))
        new_traj = []
        # if model != "DMP":
        new_traj.append(get_reproduction(model, demo=traj,\
                                         C=[np.array([x_demo[0] + 10, y_demo[0] + 10]),
                                            np.array([x_demo[-1] + 10, y_demo[-1] + 10])], \
                                         inds=[0, num_points - 1], lmbda=lmbda))
        new_traj.append(get_reproduction(model, demo=traj, \
                                         C=[np.array([x_demo[0] - 10, y_demo[0] - 10]),
                                            np.array([x_demo[-1] - 10, y_demo[-1] - 10])], \
                                         inds=[0, num_points - 1], lmbda=lmbda))
        new_traj.append(get_reproduction(model, demo=traj, \
                                         C=[np.array([x_demo[0] + 20, y_demo[0] + 20]),
                                            np.array([x_demo[20] + 20, y_demo[20] + 20]),
                                            np.array([x_demo[-1] + 20, y_demo[-1] + 20])], \
                                         inds=[0, 20, num_points - 1], lmbda=lmbda, k=0.1, D=10))

        axs[fig_x, fig_y].plot(traj[:, 0], traj[:, 1], color='black', label='original')
        a = 1
        for each in new_traj:
            axs[fig_x, fig_y].plot(each[:, 0], each[:, 1], color=colors['new_traj' + str(a)])
            a += 1

        fig_y += 1
        if fig_y >= 7:
            fig_y = 0
            fig_x += 1

    fig.legend(handles=[original_patch, new_traj1_patch, new_traj2_patch, new_traj3_patch])
    fig.suptitle(model)
    plt.savefig(filepath)
    # plt.show()


if __name__ == '__main__':
    #test("LTE", "pictures/LTE.png")
    test("DMP", "pictures/DMP.png")
    test("JA", "pictures/JA.png", 10000)

    # test("DMP")
