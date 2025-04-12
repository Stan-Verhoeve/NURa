import numpy as np
from .integrate import romberg
from .likelihoods import gaussian_logL, gaussian_logL_gradient


class GalaxyDistribution:
    """
    Theory class for galaxy distributions. Main idea is
    to hold parameters and recalculate integrals only ONCE
    when new params are passed

    Can I make it agnostic to data? Probably
    Need to make it only contain funtions that I can extract
    and the only thing that is fixed, is the params
    """

    def __init__(self, order):
        # Order of integration scheme
        self.order = order
        # Parameter vector
        self.__theta = None

        # Normalisation partition
        self.__Z = None
        self.__dZ = None

        return

    @property
    def Z(self):
        return self.__Z

    @property
    def dZ(self):
        return self.__dZ

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, new_theta):
        self.__theta = new_theta

        # Recompute normalisation integral and
        # derivative wrt parameters
        self.__Z = self.partition(*new_theta)
        self.__dZ = self.dpartition_dparams(*new_theta)

    def galaxy_dist(self, x, Nsat, a, b, c):
        """Non-normalised galaxy dist"""
        return (
            4 * np.pi * Nsat * x ** (a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))
        )

    def dgalaxy_dparam(self, x, Nsat, a, b, c, which="a"):
        """Derivative of non-normalised dist wrt its params"""
        if which == "a":
            extra_term = np.log(x / b)
        if which == "b":
            extra_term = (c * (x / b) ** c - (a - 3)) / b
        if which == "c":
            extra_term = -1 * np.log(x / b) * (x / b) ** c

        return self.galaxy_dist(x, Nsat, a, b, c) * extra_term

    def partition(self, a, b, c):
        """Normalisation partition"""
        integrand = lambda x: self.galaxy_dist(x, 1, a, b, c)

        return romberg(integrand, (1e-4, 5), m=self.order)

    def dpartition_dparams(self, a, b, c):
        """Partition derivative wrt one of its params"""
        df_da = lambda x: self.dgalaxy_dparam(x, 1, a, b, c, which="a")
        df_db = lambda x: self.dgalaxy_dparam(x, 1, a, b, c, which="b")
        df_dc = lambda x: self.dgalaxy_dparam(x, 1, a, b, c, which="c")

        dpart_da = romberg(df_da, (1e-4, 5), m=self.order)
        dpart_db = romberg(df_db, (1e-4, 5), m=self.order)
        dpart_dc = romberg(df_dc, (1e-4, 5), m=self.order)

        return [dpart_da, dpart_db, dpart_dc]

    def model(self, x, Nsat, a, b, c):
        """Normalised model"""
        return self.galaxy_dist(x, Nsat, a, b, c) / self.Z

    def dmodel_dparam(self, x, Nsat, a, b, c, which="a"):
        """Model derivative wrt one of its params"""
        if which == "a":
            extra_term = np.log(x / b)
            dZ = self.dZ[0]
        if which == "b":
            extra_term = (c * (x / b) ** c - (a - 3)) / b
            dZ = self.dZ[1]
        if which == "c":
            extra_term = -1 * np.log(x / b) * (x / b) ** c
            dZ = self.dZ[2]

        # Product rule
        return self.galaxy_dist(x, Nsat, a, b, c) * (
            extra_term / self.Z - dZ / self.Z**2
        )
        # return -1 / self.Z**2 * self.galaxy_dist(x, Nsat, a, b, c) * dZ + 1/self.Z * self.galaxy_dist(x, Nsat, a, b, c) * extra_term

    def bin_function(self, func, binedges):
        N = len(binedges) - 1
        result = np.zeros(N)

        for i in range(N):
            result[i] = romberg(func, (binedges[i], binedges[i + 1]), m=self.order)

        return result

    def binned_model(self, binedges, Nsat, a, b, c):
        func = lambda x: self.model(x, Nsat, a, b, c)
        return self.bin_function(func, binedges)

    def dmodel_dparams_binned(self, binedges, Nsat, a, b, c, which):
        dmodel_dparam = lambda x: self.dmodel_dparam(x, Nsat, a, b, c, which=which)

        return self.bin_function(dmodel_dparam, binedges)
