import numpy as np
import autograd.numpy as anp
import math
import numpy.matlib as npm
import matplotlib.pyplot as plt
import torch
from functools import partial
from autograd import elementwise_grad as egrad


def test_func(x, example, normalizer=None):

    size = x.shape
    Ns = size[0]
    dim = size[1]

    if example == 1:
        f = 0.5*(np.sin(2.0* math.pi * np.sum(x,axis=1)) + 1)
        df = npm.repmat(np.expand_dims(0.5 *2.0 * math.pi * np.cos(2.0* math.pi * np.sum(x,axis=1)), axis=1),1,dim)
        
    elif example == 2:
        x1 = np.copy(x)
        x1[:,0] = x1[:,0] - 0.5
        f = np.exp(-1.0 * np.sum(x1**2, axis=1))
        df = -2.0 * x1 * npm.repmat(np.expand_dims(f, axis=1),1,dim)

    elif example == 3:
        f = x[:,0]**3 + x[:,1]**3 + x[:,0] * 0.2 + 0.6 * x[:,1]
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        df[:,0] = 3.0*x[:,0]**2.0 + 0.2
        df[:,1] = 3.0*x[:,1]**2.0 + 0.6
        
    # wing function
    elif example == 4:
        func = partial(wing, normalizer=normalizer)
        f = func(x)
        df = egrad(func)(x)
        
    # piston function
    elif example == 5:
        func = partial(piston, normalizer=normalizer)
        f = func(x)
        df = egrad(func)(x)
        
    # circuit function
    elif example == 6:
        func = partial(circuit, normalizer=normalizer)
        f = func(x)
        df = egrad(func)(x)
        
    else:
        print('Wrong example number!')

    return (f,df)


def wing(x, normalizer=None):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns column vector of wing function at each row of inputs
    if normalizer:
        x = normalizer.inverse_transform(x)
    
    Sw = x[:,0]; Wfw = x[:,1]; A = x[:,2]; L = x[:,3]*anp.pi/180.; q = x[:,4]
    l = x[:,5]; tc = x[:,6]; Nz = x[:,7]; Wdg = x[:,8]; Wp = x[:,9]
    
    return (.036*Sw**.758*Wfw**.0035*A**.6*anp.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp)


def piston(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
        
    M = x[:, 0]
    S = x[:, 1]
    V0 = x[:, 2]
    k = x[:, 3]
    P0 = x[:, 4]
    Ta = x[:, 5]
    T0 = x[:, 6]
    
    Aterm1 = P0 * S
    Aterm2 = 19.62 * M
    Aterm3 = -k*V0 / S
    A = Aterm1 + Aterm2 + Aterm3
    
    Vfact1 = S / (2*k)
    Vfact2 = anp.sqrt(A**2 + 4*k*(P0*V0/T0)*Ta)
    V = Vfact1*(Vfact2 - A)
    
    fact1 = M
    fact2 = k + (S**2)*(P0*V0/T0)*(Ta/(V**2))
    
    C = 2 * anp.pi * anp.sqrt(fact1/fact2)
    return C


def circuit(x, normalizer=None):
    if normalizer:
        x = normalizer.inverse_transform(x)
        
    Rb1 = x[:, 0]
    Rb2 = x[:, 1]
    Rf = x[:, 2]
    Rc1 = x[:, 3]
    Rc2 = x[:, 4]
    beta = x[:, 5]
    
    Vb1 = 12*Rb2 / (Rb1+Rb2)
    term1a = (Vb1+0.74)*beta*(Rc2+9)
    term1b = beta*(Rc2+9) + Rf
    term1 = term1a / term1b
    
    term2a = 11.35 * Rf
    term2b = beta*(Rc2 + 9) + Rf
    term2 = term2a / term2b
    
    term3a = 0.74 * Rf * beta * (Rc2 + 9)
    term3b = (beta*(Rc2 + 9) + Rf) * Rc1
    term3 = term3a / term3b
    
    Vm = term1 + term2 + term3
    
    return Vm


def gridplot(grid_np, Nx=64, Ny=64, color='black', **kwargs):
    grid_1 = grid_np[:, 0].reshape(1, 1, Nx, Ny)
    grid_2 = grid_np[:, 1].reshape(1, 1, Nx, Ny)
    u = np.concatenate((grid_1, grid_2), axis=1)
    
    # downsample displacements
    h = np.copy(u[0, :, ::u.shape[2]//Nx, ::u.shape[3]//Ny])
    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0, ...] /= float(u.shape[2])/Nx
    h[1, ...] /= float(u.shape[3])/Ny
    # put back into original index space
    h[0, ...] *= float(u.shape[2])/Nx
    h[1, ...] *= float(u.shape[3])/Ny
    
    plt.figure(figsize=(6, 4))
    # create a meshgrid of locations
    for i in range(Nx):
        plt.plot(h[0, i, :], h[1, i, :], color=color, **kwargs)
    for i in range(Ny):
        plt.plot(h[0, :, i], h[1, :, i], color=color, **kwargs)
    for ix, xn in zip([0, -1], ['B', 'T']):
        for iy, yn in zip([0, -1], ['L', 'R']):
            plt.plot(h[0, ix, iy], h[1, ix, iy], 'o', label='({xn},{yn})'.format(xn=xn, yn=yn))
    
    plt.axis('equal')
    plt.legend()
    plt.grid(linestyle='dotted')
    plt.show()
    
    
def as_sensitivity(ss, x, example, normalizer=None):

    size = x.shape
    Ns = size[0]
    dim = size[1]
    dx = 0.001

    sen_ind_new = np.zeros(dim)
    sen_ind_old = np.zeros(dim)

    # Transformed x
    f1,df1 = test_func(x, example, normalizer)
    x1 = ss.transform(x)

    for i in range(0,dim):

        # Perturb x1 in the transformed space
        y1 = np.copy(x1[0])
        z1 = np.copy(x1[1])
        x2 = np.hstack((y1, z1))
            
        x2[:,i] = x2[:,i] + dx
        y2 = x2[:, 0:ss.dim]
        z2 = x2[:, ss.dim:]
        
        lc = np.dot(ss.W1, y2.transpose()) + np.dot(ss.W2, z2.transpose())
        f2,df2 = test_func(lc.transpose(), example, normalizer)
        sen_ind_new[i] = np.mean(np.abs(f2-f1)/dx)

        x3 = np.copy(x)
        x3[:,i] = x3[:,i] + dx
        f3,df3 = test_func(x3, example, normalizer)
        sen_ind_old[i] = np.mean(np.abs(f3-f1)/dx)


    return sen_ind_old,sen_ind_new     
    
    
def nll_sensitivity(nll, x, example, normalizer=None):

    size = x.shape
    Ns = size[0]
    dim = size[1]
    dx = 0.001

    sen_ind_new = np.zeros(dim)
    sen_ind_old = np.zeros(dim)

    # Transformed x
    f1,df1 = test_func(x,example, normalizer)
    x1 = nll.forward(torch.from_numpy(x))

    for i in range(0,dim):

        # Perturb x1 in the transformed space
        x2 = torch.clone(x1)
        x2[:,i] = x2[:,i] + dx
        f2,df2 = test_func(nll.backward(x2).detach().numpy(), example, normalizer)
        sen_ind_new[i] = np.mean(np.abs(f2-f1)/dx)

        x3 = np.copy(x)
        x3[:,i] = x3[:,i] + dx
        f3,df3 = test_func(x3, example, normalizer)
        sen_ind_old[i] = np.mean(np.abs(f3-f1)/dx)

    return sen_ind_old,sen_ind_new
