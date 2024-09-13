import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor


def generate_belief_matrix(grid_size: int, margin_size: int, X: np.ndarray, y: np.ndarray,
                           gpc: GaussianProcessClassifier, class_index: int = 1):
    gpc.fit(X, y)
    size = grid_size + margin_size
    x = np.meshgrid(range(size), range(size))
    s_prob = gpc.predict_proba(np.array(x).reshape(2, -1).T)
    return s_prob[:, class_index].reshape(size, size)[
           margin_size // 2:-margin_size // 2,
           margin_size // 2:-margin_size // 2]


def generate_belief_mean_matrix(grid_size: int, gpr: GaussianProcessRegressor, return_std=True):
    x = np.meshgrid(range(grid_size), range(grid_size))
    if return_std:
        s_mean, s_std = gpr.predict(np.array(x).reshape(2, -1).T, return_std=True)
        return s_mean.reshape(grid_size, grid_size), s_std.reshape(grid_size, grid_size)
    else:
        s_mean, s_cov = gpr.predict(np.array(x).reshape(2, -1).T, return_std=False, return_cov=True)
        return s_mean.reshape(grid_size, grid_size), s_cov


def construct_dataset_info(grid_size: int, margin_size: int, locs: np.ndarray, step_size: int = 20):
    # make a copy of the agent locations
    locs = locs.copy()

    # add margin to agent locations
    locs += margin_size // 2

    # meshgrid size
    mesh_size = (grid_size + margin_size) // step_size
    mesh = np.array(np.meshgrid(range(mesh_size), range(mesh_size))) * step_size

    # mesh to long format
    mesh = mesh.reshape(2, -1).T

    # remove mesh point if it is too close to the agent
    dist = np.linalg.norm(locs[:, None] - mesh, axis=2)
    mask = np.min(dist, axis=0) > step_size
    mesh = mesh[mask]

    # combine agent locations and mesh
    X = np.vstack([locs, mesh])
    y = np.hstack([np.ones(locs.shape[0]), np.zeros(mesh.shape[0])])

    return X, y
