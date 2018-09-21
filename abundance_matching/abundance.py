from astropy import cosmology
import astropy.units as u
import numpy as np
import os
from scipy import integrate


def _get_data_path(data_file):
    """Returns the path of the data file in the subdirectory.

    We know the relative path of it compared to this file, so it's easy to
    know where it is."""
    this_file_dir = os.path.dirname(__file__)
    return this_file_dir + "/data/{}".format(data_file)

def a_to_z(a):
    return (1.0 / a) - 1.0

def z_to_a(z):
    return 1.0 / (1.0 + z)

def z_to_age(z):
    ages = [cosmology.Planck15.age(z_i).to("Myr") for z_i in z]
    return u.Quantity(ages)

class Behroozi13(object):
    def __init__(self):
        self.load_stellar_mass_hui()
        self.load_sfh()

    # def load_stellar_mass(self):
    #     file = _get_data_path("behroozi_13_data/sm/sm_hist_rel_12.0.dat")
    #
    #     m_star_0 = (10**10.428106376900988) * u.solMass  # from Hui's code
    #
    #     a, m_m_0, err_up, err_down = [], [], [], []
    #     with open(file, "r") as in_file:
    #         for line in in_file:
    #             split_line = line.split()
    #
    #             a.append(float(split_line[0]))
    #             m_m_0.append(float(split_line[1]))
    #             err_up.append(float(split_line[2]))
    #             err_down.append(float(split_line[3]))
    #
    #     m_m_0 = np.array(m_m_0)
    #     err_up = np.array(err_up)
    #     err_down = np.array(err_down)
    #
    #     self.m_star_mean = m_star_0 * m_m_0
    #     self.m_star_low_bound = m_star_0 * (m_m_0 - err_down)
    #     self.m_star_up_bound = m_star_0 * (m_m_0 + err_up)
    #
    #     self.m_star_a = np.array(a)
    #     self.m_star_z = a_to_z(self.m_star_a)
    #     self.m_star_ages = z_to_age(self.m_star_z)

    def load_stellar_mass_hui(self):
        file = _get_data_path("z_Ms.txt")

        z, mean, up_bound, low_bound = [], [], [], []
        with open(file, "r") as in_file:
            for line in in_file:
                split = line.split()

                z.append(float(split[0]))
                mean.append(float(split[1]))
                up_bound.append(float(split[2]))
                low_bound.append(float(split[3]))

        self.m_star_z = np.array(z)
        self.m_star_a = z_to_a(self.m_star_z)
        self.m_star_ages = z_to_age(self.m_star_z)
        self.m_star_mean = u.Quantity(mean, u.solMass)
        self.m_star_up_bound = u.Quantity(up_bound, u.solMass)
        self.m_star_low_bound = u.Quantity(low_bound, u.solMass)

    def load_sfh(self):
        """Get the sfh for the 10^12 halo"""
        file = _get_data_path("behroozi_13_data/sfh/sfh_z0.1_corrected_12.0.dat")
        with open(file, "r") as in_file:
            a, sfh, err_up, err_down = [], [], [], []
            for line in in_file:
                this_a, this_sfh, this_err_up, this_err_down = line.split()
                a.append(float(this_a))
                sfh.append(float(this_sfh))
                err_up.append(float(this_err_up))
                err_down.append(float(this_err_down))

        self.sfh = u.Quantity(sfh, u.solMass / u.year)
        self.sfh_err_up = u.Quantity(err_up, u.solMass / u.year)
        self.sfh_err_down = u.Quantity(err_down, u.solMass / u.year)

        self.sfh_up_bound = self.sfh + self.sfh_err_up
        self.sfh_low_bound = self.sfh - self.sfh_err_down

        self.sfh_a = np.array(a)
        self.sfh_z = a_to_z(self.sfh_a)
        self.sfh_ages = z_to_age(self.sfh_z)

class UniverseMachine(object):
    def __init__(self):
        self.read_sfh()

    def read_sfh(self):
        file = _get_data_path("universe_machine/data/sfhs/sfh_hm12.00_a1.002310.dat")
        with open(file, "r") as in_file:
            a = []
            sfh_all, err_up_all, err_down_all = [], [], []
            sfh_cen, err_up_cen, err_down_cen = [], [], []
            sfh_sat, err_up_sat, err_down_sat = [], [], []

            for line in in_file:
                if line.startswith("#"):
                    continue
                split_line = line.split()

                a.append(float(split_line[0]))

                sfh_all.append(float(split_line[1]))
                err_up_all.append(float(split_line[2]))
                err_down_all.append(float(split_line[3]))

                sfh_cen.append(float(split_line[10]))
                err_up_cen.append(float(split_line[11]))
                err_down_cen.append(float(split_line[12]))

                sfh_sat.append(float(split_line[13]))
                err_up_sat.append(float(split_line[14]))
                err_down_sat.append(float(split_line[15]))

        self.sfh_a = np.array(a)
        self.sfh_z = a_to_z(self.sfh_a)
        self.ages = z_to_age(self.sfh_z)

        self.sfh_all = u.Quantity(sfh_all, "Msun/year")
        self.sfh_err_up_all = u.Quantity(err_up_all, "Msun/year")
        self.sfh_err_down_all = u.Quantity(err_down_all, "Msun/year")
        self.sfh_up_bound_all = self.sfh_all + self.sfh_err_up_all
        self.sfh_low_bound_all = self.sfh_all - self.sfh_err_down_all

        self.sfh_cen = u.Quantity(sfh_cen, "Msun/year")
        self.sfh_err_up_cen = u.Quantity(err_up_cen, "Msun/year")
        self.sfh_err_down_cen = u.Quantity(err_down_cen, "Msun/year")
        self.sfh_up_bound_cen = self.sfh_cen + self.sfh_err_up_cen
        self.sfh_low_bound_cen = self.sfh_cen - self.sfh_err_down_cen

        self.sfh_sat = u.Quantity(sfh_sat, "Msun/year")
        self.sfh_err_up_sat = u.Quantity(err_up_sat, "Msun/year")
        self.sfh_err_down_sat = u.Quantity(err_down_sat, "Msun/year")
        self.sfh_up_bound_sat = self.sfh_sat + self.sfh_err_up_sat
        self.sfh_low_bound_sat = self.sfh_sat - self.sfh_err_down_sat

