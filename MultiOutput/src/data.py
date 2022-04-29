import numpy as np
import math

def make_monte_carlo(means, 
                     sigma_y=None, 
                     size=1000,
                     n_noise=2, 
                     components_size=None, 
                     n_y=2, 
                     componenets_y=None, 
                     components_y_size=None, 
                     return_lables=False, 
                     y_noise_level=0.5):
    n_components = means.shape[0]
    
    mean_noise = np.ones(n_noise)
    if components_size is None:
        components_size = np.ones(n_components)/n_components
    else:
        assert len(components_size) == n_components
    sizes_x = [math.ceil(size*components_size[i]) for i in range(n_components-1)]
    sizes_x.append(size - sum(sizes_x))
    
    X_pure = np.concatenate([
                    np.random.multivariate_normal(
                        means[i], np.diag(np.ones_like(means[i])), size=sizes_x[i]
                    ) for i in range(n_components)
                ])
    X_noise = np.random.multivariate_normal(mean_noise, np.diag(np.ones(n_noise)), X_pure.shape[0])
    X = np.concatenate([X_pure, X_noise], axis=1)

    if componenets_y is None:
        componenets_y = n_components
        
    if components_y_size is None:
        components_y_size = np.ones(componenets_y)/componenets_y
    else:
        assert len(n_components_y_size) == n_componenets_y
        
    sizes_y = [math.ceil(size*components_y_size[i]) for i in range(componenets_y-1)]
    sizes_y.append(size - sum(sizes_y))
    
    if sigma_y is None:
        l_y = np.concatenate(
            [
                np.tril(np.random.random((n_y,n_y))*y_noise_level)[None, :] for i in range(componenets_y)
            ])
        sigma_y = np.concatenate([
            (l@l.T)[None, :] for l in l_y
        ])
    else:
        for sigma in sigma_y:
            assert sigma.shape[0] == n_y
       
    y_eps = np.concatenate([
            np.random.multivariate_normal(np.zeros(n_y), sigma_y[i], size=sizes_y[i]) for i in range(componenets_y)
        ])
    y_l = np.concatenate([
        np.repeat(l_y[i][np.tril_indices(n_y)][None, :], sizes_y[i], axis=0) for i in range(componenets_y)
    ])
    y_means = np.concatenate([
            np.ones((sizes_x[i], n_y)) + i*1
            for i in range(n_components)
        ])
    
    Y = y_means + y_eps
    return X, Y, y_means, y_l
    
    
def get_positive_definite(n):
        L = np.tril(np.random.random((n_y,n_y)))
        return L@L.T
    