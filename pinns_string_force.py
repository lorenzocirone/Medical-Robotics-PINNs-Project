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


epochs = 20000


def func(x):
    x, t = np.split(x, 2, axis=1)

    return w1(x)


def w1(x):

    return 0


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


def go():

    sig = 50
    C = 10
    print(f"Start training C={C} sig={sig}")

    name = f"pinns_string_force"
    script_directory = pathlib.Path.cwd() / f"{name}"
    if not script_directory.exists():
        os.makedirs(script_directory, exist_ok=True)

    model_dir = os.path.join(script_directory, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    def f(sample):
        x = sample[:, 0]
        x_f = 0.8
        height = 1
        t = sample[:, -1]

        alpha = 8.9
        za = -height * tf.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
        return za

    tf.reset_default_graph()
    def pde(x, y):
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)

        # Uncomment the following to consider stiffness
        # dy_xxxx = dde.grad.hessian(dy_xx, x, i=0, j=0)
        return dy_tt - C ** 2 * dy_xx  - f(x)# + ESK2 * dy_xxxx
    
    sig = 50
    C = 10

    def summative(x, n_max=3):
         x, t = np.split(x, 2, axis=1)
         integrals = [compute_integral(n) for n in range(1, n_max + 1)]
         integrals2 = [compute_integral2(n) for n in range(1, n_max + 1)]
         terms = [
    	    ((2) * integrals[n - 1] * np.cos(n * np.pi * C * t) + (2 / (n * np.pi * C)) * integrals2[n - 1] * np.sin(
    	        n * np.pi * C * t)) * np.sin(n * np.pi * x) for n in range(1, n_max + 1)]
         return np.sum(terms, axis=0)


    def plot_animation(c, nume):
        filename = f"{script_directory}/animation_{nume}.gif"
        dx = 0.01  # Spatial step
        dt = 0.01  # Time step
        L = 1
        t_max = 1  # Maximum time

        # Discretization
        x_values = np.arange(0, L, dx)
        t_values = np.arange(0, t_max, dt)

        p = np.zeros((len(x_values), len(t_values)))

        # Set initial condition
        XX = np.vstack((x_values, np.zeros_like(x_values))).T
        p[:, 0] = c.predict(XX).reshape(len(XX), )

        # Update function using exact solution
        def update(frame):
            x = x_values.reshape(-1, 1)
            t = t_values[frame]
            XX = np.vstack((x_values, np.full_like(x_values, t))).T

            p[:, frame] = c.predict(XX).reshape(len(XX), ).flatten()

            pred_line.set_ydata(p[:, frame])

            plt.xlabel('x')
            plt.ylabel('Amplitude')
            plt.title(f'Wave Equation Animation at t={t:.2f}')
            plt.legend()  # Show legend with labels
            return pred_line

        # Create the animation
        fig, ax = plt.subplots()
        pred_line, = ax.plot(x_values, p[:, 0], label='Predicted')  # New line

        ax.set_ylim(-1, 1)  # Adjust the y-axis limits if needed

        ani = FuncAnimation(fig, update, frames=len(t_values), interval=1.0, blit=True)

        plt.show()

        # Save the animation as a GIF file
        ani.save(filename, writer='imagemagick')


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
        [ic_1, ic_2],
        num_domain=1440,
        num_boundary=360,
        num_initial=360,
        num_test=10000,
    )

    layer_size = [2] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.STMsFFN(
        layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, sig]
    )


    net.apply_output_transform(lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * y)


    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=0.001,
        decay=("inverse time", 2000, 0.9),
    )
    pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)
    losshistory, train_state = model.train(
        iterations=epochs, callbacks=[pde_residual_resampler, early_stopping], display_every=500, model_save_path=f"{model_dir}/{name}"
    )
    dde.saveplot(losshistory, train_state, output_dir=f"{script_directory}")
    # model.restore(f"{model_dir}/{name}-{epochs}.ckpt", verbose=0)


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

    # Plot
    U_pred = u_pred.reshape(100, 100)

    # Predictions
    plt.figure()
    plt.pcolor(t, x, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predicted u(x)')
    plt.savefig(f"{script_directory}/dde1_{name}.png")
    
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
    plt.savefig(f"{script_directory}/dde3_{name}.png")
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

    # plot_animation(model, name)

if __name__ == "__main__":

    go()