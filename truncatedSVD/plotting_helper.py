import numpy as np
import matplotlib.pyplot as plt

def plot_residual_norms(errs, filename, title="Residual Norms for Singular Vectors"):
    position = np.arange(errs.shape[0])
    plt.figure()
    plt.plot(position,errs)
    plt.title(title)
    plt.ylabel("Residual Norm")
    plt.yscale('log')
    plt.ylim(0.0001,2)
    plt.yticks([1e0,1e-2,1e-4])    
    plt.xlabel("Singular Value Index")
    plt.savefig("../figures/"+filename+".pdf")

def plot_relative_errs(errs, filename, title="Relative Error for Singular Vectors"):
    position = np.arange(errs.shape[0])
    plt.figure()
    plt.plot(position,errs)
    plt.title(title)
    plt.yscale('log')
    plt.ylim(0.0000000001,1)
    plt.yticks([1e0,1e-5,1e-10])     
    plt.ylabel("Relative Error")
    plt.xlabel("Singular Value Index")
    plt.savefig("../figures/"+filename+".pdf")


def plot_stacked_residual_norms(errs1, errs2, errs3, filename, title="Residual Norms for Singular Vectors"):
    position = np.arange(errs1.shape[0])
    plt.figure()
    plt.plot(position,errs1,label="$\phi=1$")
    plt.plot(position,errs2,label="$\phi=6$")
    plt.plot(position,errs3,label="$\phi=12$")
    plt.title(title)
    plt.ylabel("Residual Norm")
    plt.yscale('log')
    plt.ylim(0.0001,2)
    plt.yticks([1e0,1e-2,1e-4])    
    plt.xlabel("Singular Value Index")
    plt.legend()
    plt.savefig("../figures/"+filename+".pdf")

def plot_stacked_relative_errs(errs1, errs2, errs3, filename, title="Relative Error for Singular Vectors"):
    position = np.arange(errs1.shape[0])
    plt.figure()
    plt.plot(position,errs1,label="$\phi=1$")
    plt.plot(position,errs2,label="$\phi=6$")
    plt.plot(position,errs3,label="$\phi=12$")
    plt.title(title)
    plt.yscale('log')
    plt.ylim(0.0000000001,1)
    plt.yticks([1e0,1e-5,1e-10])     
    plt.ylabel("Relative Error")
    plt.xlabel("Singular Value Index")
    plt.legend()
    plt.savefig("../figures/"+filename+".pdf")
