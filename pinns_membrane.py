
# -*- coding: utf-8 -*-

"""Backend supported: tensorflow.compat.v1, paddle

Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import deepxde as dde
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


A = 2
ESK2 = 0.1
C = 5

epochs = 40000


def get_initial_loss(model):
    model.compile("adam", lr=0.001,
                  # metrics=["l2 relative error"]
                  )
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


def func(x):
    x, y, t = tf.split(x, 3, axis=1)

    return w1(x)


def func2(x):
    x, y, t = tf.split(x, 3, axis=1)

    return w2(x, y)


def w1(x):
    # sigma = 0.05
    # return np.exp(-(x - 0.23)**2 / (2 * sigma**2))
    # condition = tf.math.less_equal(x, 0.2)
    # return tf.where(condition, 5 * x, 1.25 * (1 - x))
    # condition = np.less_equal(x, 0.2)
    # return np.where(condition, 5 * x, 1.25 * (1 - x))
    return 0

def w2(x, y):

    return 500000 * tf.exp(-(x - 0.5)**2 / 0.05) * tf.exp(-(y - 0.5)**2 / 0.05)


def plot_animation(c, nume):
    filename = f"{nume}/animation.gif"

    dt = 0.01  # Time step
    t_max = 1  # Maximum time

    # Generate meshgrid for x and y values
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x_vals, y_vals)
    t_values = np.arange(0, t_max, dt)

    p = np.zeros((len(x_vals), len(y_vals), len(t_values)))

    # Set initial condition
    XX = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten()))).T
    p[:, :, 0] = c.predict(XX).reshape(len(x_vals), len(y_vals))

    # Update function using predicted solution
    def update(frame):
        t = t_values[frame]
        XX = np.vstack((x.flatten(), y.flatten(), np.full_like(x.flatten(), t))).T

        p[:, :, frame] = c.predict(XX).reshape(len(x_vals), len(y_vals))

        ax.clear()
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Deformation (Z-axis)')
        ax.set_title(f'Membrane Deformation at t={t:.2f}')

        # Plot the 3D surface without colormap
        surf = ax.plot_surface(x, y, p[:, :, frame], rstride=1, cstride=1, alpha=0.8, antialiased=True)

        # Set fixed z-axis limits
        ax.set_zlim(-1, 1)
        return surf,

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = FuncAnimation(fig, update, frames=len(t_values), interval=1.0, blit=True)

    # Save the animation as a GIF file
    ani.save(filename, writer='imagemagick')


def compute_integral(n):
    integrand = lambda z: w1(z) * np.sin(n * np.pi * z)
    result, _ = quad(integrand, 0, 1)
    return result


def compute_integral2(n):
    integrand = lambda z: w2(z) * np.sin(n * np.pi * z)
    result, _ = quad(integrand, 0, 1)
    return result



def go():


    print(f"Start training membrane")

    name = f"results"
    script_directory = pathlib.Path.cwd() / f"{name}"
    if not script_directory.exists():
        os.makedirs(script_directory, exist_ok=True)

    tf.reset_default_graph()
    def pde(x, z):
        dz_tt = dde.grad.hessian(z, x, i=2, j=2)
        dz_xx = dde.grad.hessian(z, x, i=0, j=0)
        dz_yy = dde.grad.hessian(z, x, i=1, j=1)

        # Uncomment the following to consider stiffness
        # dy_xxxx = dde.grad.hessian(dy_xx, x, i=0, j=0)
        return dz_tt - C ** 2 *(dz_xx+dz_yy)  # + ESK2 * dy_xxxx


    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
    # do not use dde.NeumannBC here, since `normal_derivative` does not work with temporal coordinate.
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1) - func2(x),
        lambda x, _: np.isclose(x[1], 0),
    )
    data = dde.data.TimePDE(
        geomtime,
        pde,
        # [bc, ic_1, ic_2],
        [ic_1, ic_2],
        # [ic_2],
        num_domain=1440,
        num_boundary=360,
        num_initial=360,
        # solution=summative,
        num_test=10000,
    )

    layer_size = [3] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.STMsFFN(
        layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
    )


    net.apply_output_transform(lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2]) * y)


    model = dde.Model(data, net)
    initial_losses = get_initial_loss(model)
    loss_weights = len(initial_losses) / initial_losses
    model.compile(
        "adam",
        lr=0.001,
        # metrics=["l2 relative error"],
        loss_weights=loss_weights,
        decay=("inverse time", 2000, 0.9),
    )
    pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)
    losshistory, train_state = model.train(
        iterations=epochs, callbacks=[pde_residual_resampler, early_stopping], display_every=500, model_save_path=f"{script_directory}/{name}"
    )
    dde.saveplot(losshistory, train_state, output_dir=f"{script_directory}")


    # Predictions
    x = np.linspace(0, 1, num=100)
    y = np.linspace(0, 1, num=100)
    x, y = np.meshgrid(x, y)
    X_0 = np.hstack((x.flatten()[:, None], y.flatten()[:, None], np.zeros_like(x.flatten()[:, None])))
    X_025 = np.hstack((x.flatten()[:, None], y.flatten()[:, None], np.full_like(x.flatten()[:, None], 0.25)))
    X_05 = np.hstack((x.flatten()[:, None], y.flatten()[:, None], np.full_like(x.flatten()[:, None], 0.5)))
    X_075 = np.hstack((x.flatten()[:, None], y.flatten()[:, None], np.full_like(x.flatten()[:, None], 0.75)))
    X_1 = np.hstack((x.flatten()[:, None], y.flatten()[:, None], np.ones_like(x.flatten()[:, None])))

    U_0 = model.predict(X_0).reshape(100, 100)
    U_025 = model.predict(X_025).reshape(100, 100)
    U_05 = model.predict(X_05).reshape(100, 100)
    U_075 = model.predict(X_075).reshape(100, 100)
    U_1 = model.predict(X_1).reshape(100, 100)


    # Predictions
    fig = plt.figure(5, figsize=(30, 5))

    # Loop over the time steps
    for i, (U, t) in enumerate([(U_0, 0), (U_025, 0.25), (U_05, 0.5), (U_075, 0.75), (U_1, 1.0)]):
        plt.subplot(1, 5, i + 1)
        plt.pcolor(x, y, U, cmap='jet')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title(f'Prediction at t={t}')

        # Set axis limits to range from 0 to 1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{script_directory}/dde1_{name}.png")
    plt.show()

    # # Convert the list of arrays to a 2D NumPy array
    matrix = np.array(losshistory.loss_train)
    #
    # # Separate the components into different arrays
    loss_res = matrix[:, 0]
    # # loss_bcs = matrix[:, 1]
    loss_u_t_ics = matrix[:, 1]
    loss_du_t_ics = matrix[:, 2]
    #
    # l2_error = np.array(losshistory.metrics_test)
    #
    fig = plt.figure(figsize=(6, 5))
    iters = 500 * np.arange(len(loss_res))
    with sns.axes_style("darkgrid"):
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$')
        # plt.plot(iters, loss_bcs, label='$\mathcal{L}_{u}$')
        plt.plot(iters, loss_u_t_ics, label='$\mathcal{L}_{u_0}$')
        plt.plot(iters, loss_du_t_ics, label='$\mathcal{L}_{u_t}$')
        # plt.plot(iters, l2_error, label='$\mathcal{L}^2 error$')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{script_directory}/dde2_{name}.png")
        plt.show()

    plot_animation(model, script_directory)

if __name__ == "__main__":
    go()