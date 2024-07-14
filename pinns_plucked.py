
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


#epochs = 1000000
epochs = 10000


def get_initial_loss(model):
    model.compile("adam", lr=0.001,
                  metrics=["l2 relative error"]
                  )
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


def func(x):
    x, t = np.split(x, 2, axis=1)

    return w1(x)


def w1(x):
    # sigma = 0.05
    # return np.exp(-(x - 0.23)**2 / (2 * sigma**2))
    # condition = tf.math.less_equal(x, 0.2)
    # return tf.where(condition, 5 * x, 1.25 * (1 - x))
    condition = np.less_equal(x, 0.2)
    return np.where(condition, 5 * x, 1.25 * (1 - x))


def w2(z):

    return 0


def compute_integral(n):
    integrand = lambda z: w1(z) * np.sin(n * np.pi * z)
    result, _ = quad(integrand, 0, 1)
    return result


def compute_integral2(n):
    integrand = lambda z: w2(z) * np.sin(n * np.pi * z)
    result, _ = quad(integrand, 0, 1)
    return result

sig = 50
C = 10

def pde(x, y):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    # Uncomment the following to consider stiffness
    # dy_xxxx = dde.grad.hessian(dy_xx, x, i=0, j=0)
    return dy_tt - C ** 2 * dy_xx  # + ESK2 * dy_xxxx


def summative(x, n_max=3):
    x, t = np.split(x, 2, axis=1)

    integrals = [compute_integral(n) for n in range(1, n_max + 1)]
    integrals2 = [compute_integral2(n) for n in range(1, n_max + 1)]

    terms = [
        ((2) * integrals[n - 1] * np.cos(n * np.pi * C * t) + (2 / (n * np.pi * C)) * integrals2[n - 1] * np.sin(
            n * np.pi * C * t)) * np.sin(n * np.pi * x) for n in range(1, n_max + 1)]
    return np.sum(terms, axis=0)

def create_model():
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
    # do not use dde.NeumannBC here, since `normal_derivative` does not work with temporal coordinate.
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda x, _: np.isclose(x[1], 0),
    )
    data = dde.data.TimePDE(
        geomtime,
        pde,
        # [bc, ic_1, ic_2],
        [ic_1, ic_2],
        # [ic_2],
        num_domain=1440, #  is the number of training residual points sampled inside the domain
        num_boundary=360,# number of training points sampled on the boundary
        num_initial=360, #  is the number of training residual points initially sampled
        solution=summative, # is the reference solution to compute the error of our solution
        num_test=10000,  # 10000 point to test the PDE residual 
    )

    layer_size = [2] + [100] * 3 + [1]  # 2 input, 3 hidden layer (with width 100), 1 output so the depth is 4
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.STMsFFN(
        layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, sig]
    )

    net.apply_output_transform(lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * y)

    model = dde.Model(data, net)
    initial_losses = get_initial_loss(model)
    loss_weights = len(initial_losses) / initial_losses
    model.compile(
        "adam",
        lr=0.001,
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
        decay=("inverse time", 2000, 0.9),
    )
    return model

def train():
    print(f"Start training C={C} sig={sig}")
    name = f"sig{sig}_C{C}"
    script_directory = pathlib.Path.cwd() / f"{name}"
    if not script_directory.exists():
        os.makedirs(script_directory, exist_ok=True)

    tf.reset_default_graph()

    def plot_animation(c, nume):
        filename = f"{script_directory}/animation_{nume}.gif"
        dx = 0.01  # Spatial step
        dt = 0.01  # Time step
        L = 1
        t_max = 1  # Maximum time

        # Discretization
        x_values = np.arange(0, L, dx)
        t_values = np.arange(0, t_max, dt)

        u = np.zeros((len(x_values), len(t_values)))
        p = np.zeros((len(x_values), len(t_values)))

        # Set initial condition
        u[:, 0] = w1(x_values)
        XX = np.vstack((x_values, np.zeros_like(x_values))).T
        p[:, 0] = c.predict(XX).reshape(len(XX), )

        # Update function using exact solution
        def update(frame):
            x = x_values.reshape(-1, 1)
            t = t_values[frame]
            xt_mesh = np.concatenate([x, np.full_like(x, t)], axis=1)
            XX = np.vstack((x_values, np.full_like(x_values, t))).T

            u[:, frame] = summative(xt_mesh).flatten()
            p[:, frame] = c.predict(XX).reshape(len(XX), ).flatten()

            line.set_ydata(u[:, frame])
            pred_line.set_ydata(p[:, frame])

            plt.xlabel('x')
            plt.ylabel('Amplitude')
            plt.title(f'Wave Equation Animation at t={t:.2f}')
            plt.legend()  # Show legend with labels
            return line, pred_line

        # Create the animation
        fig, ax = plt.subplots()
        line, = ax.plot(x_values, u[:, 0], label='Exact')  # Existing line
        pred_line, = ax.plot(x_values, p[:, 0], label='Predicted')  # New line

        ax.set_ylim(-1, 1)  # Adjust the y-axis limits if needed

        ani = FuncAnimation(fig, update, frames=len(t_values), interval=1.0, blit=True)

        plt.show()

        # Save the animation as a GIF file
        ani.save(filename, writer='imagemagick')


    model = create_model()
    
    pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)
    losshistory, train_state = model.train(
        iterations=epochs, callbacks=[pde_residual_resampler, early_stopping], display_every=500, model_save_path=f"{script_directory}/{name}"
    )
    dde.saveplot(losshistory, train_state, output_dir=f"{script_directory}")


    # Predictions
    t = np.linspace(0, 1, num=100)
    x = np.linspace(0, 1, num=100)
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    u_pred = model.predict(X_star)
    u_star = summative(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    print('Relative L2 error_u: %e' % (error_u))

    # Plot
    U_star = u_star.reshape(100, 100)
    U_pred = u_pred.reshape(100, 100)

    # Predictions
    fig = plt.figure(3, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(t, x, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact u(x)')

    plt.subplot(1, 3, 2)
    plt.pcolor(t, x, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predicted u(x)')

    plt.subplot(1, 3, 3)
    plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(f"{script_directory}/dde1_{name}.png")
    plt.show()

    # Convert the list of arrays to a 2D NumPy array
    matrix = np.array(losshistory.loss_train)

    # Separate the components into different arrays
    loss_res = matrix[:, 0]
    # loss_bcs = matrix[:, 1]
    loss_u_t_ics = matrix[:, 1]
    loss_du_t_ics = matrix[:, 2]

    l2_error = np.array(losshistory.metrics_test)

    fig = plt.figure(figsize=(6, 5))
    iters = 500 * np.arange(len(loss_res))
    with sns.axes_style("darkgrid"):
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$')
        # plt.plot(iters, loss_bcs, label='$\mathcal{L}_{u}$')
        plt.plot(iters, loss_u_t_ics, label='$\mathcal{L}_{u_0}$')
        plt.plot(iters, loss_du_t_ics, label='$\mathcal{L}_{u_t}$')
        plt.plot(iters, l2_error, label='$\mathcal{L}^2 error$')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{script_directory}/dde2_{name}.png")
        plt.show()

    plot_animation(model, name)



saved_model = None
def load_model(path):
    global saved_model
    if saved_model == None:
        saved_model = create_model()
        saved_model.restore(path, verbose = 1)
        return True
    else:
        return False

def predict(x_values, t_values):
    global saved_model
    if saved_model == None:
        return
    
    x = x_values.reshape(-1, 1)
    t = t_values
    xt_mesh = np.concatenate([x, np.full_like(x, t)], axis=1)
    XX = np.vstack((x_values, np.full_like(x_values, t))).T

    p = saved_model.predict(XX).reshape(len(XX), ).flatten()
    return p



if __name__ == "__main__":
    train()