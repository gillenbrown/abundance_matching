from astropy import cosmology
import astropy.units as u
from astropy import table
import numpy as np
import os
import re
import collections

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

def find_a(filename):
    # Use a regular expression
    scale_re = re.compile("""a    # single a
                             [01]  # either 0 or 1 (start of scale factor)
                             \.    # single period
                             \d{6,6}   # 6 digits
                             [_\.] # either . or _, indicating new part of name
                             """, re.VERBOSE)
    # We get the 0th index since there will be only one match per file, and
    # throw away the first and last characters (a and _.)
    return re.findall(scale_re, filename)[0][1:-1]

def find_m(filename):
    # Use a regular expression
    mass_re = re.compile("""[sh]      # either stellar or halo mass
                            m         # single m
                            \d{1,2}  # 1 or two digits for first part of mass
                            \.        # single period
                            \d{2,2}  # 2 digits for frational part of mass
                            [_\.] # either . or _, indicating new part of name
                            """, re.VERBOSE)
    # We get the 0th index since there will be only one match per file, and
    # throw away the first two and last characters (sm and _.)
    stripped_m = re.findall(mass_re, filename)[0][2:-1]
    # we want them all to have the same number of digits, to add a zero if the
    # initial digit is less than 10
    if stripped_m[0] != "1":
        return "0" + stripped_m
    else:
        return stripped_m

def find_closest(item, list_of_items):
    # we assume the original list is strings, and is sorted
    # first turn to floats
    processed_list = [float(i) for i in list_of_items]

    # first to some edge cases
    if item < processed_list[0]:
        return list_of_items[0]
    elif item > processed_list[-1]:
        return list_of_items[-1]
    else:  # somewhere in the middle
        for idx in range(len(processed_list) - 1):
            if processed_list[idx] <= item < processed_list[idx + 1]:
                # we have the bounding scale factors
                diff_1 = abs(item - processed_list[idx])
                diff_2 = abs(item - processed_list[idx + 1])
                if diff_1 < diff_2:
                    return list_of_items[idx]
                else:
                    return list_of_items[idx + 1]


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
    data_dir = _get_data_path("universe_machine/data/")
    sfh_dir = data_dir + "sfhs/"
    smhm_dir = data_dir + "smhm/median_raw/"

    def __init__(self, m_halo, m_stellar, z):
        self.m_halo = m_halo
        self.m_stellar = m_stellar
        self.z = z
        self.a = z_to_a(self.z)
        self.read_sfh()
        self.read_smhm()

    def read_smhm(self):
        # According to the readme file, the direct SMHM measurements are in the 
        # data/smhm/median_raw directory. Within this directory, the median
        # values are in the `sm_hm_a*` files, while the scatter is in the
        # `smhm_scatter_a*` files.

        smhm_files = [f for f in os.listdir(self.smhm_dir)
                      if f.startswith("smhm_a")]

        # create dictionaries to store the eventual tables
        self.smhm_table = dict()
        self.smhm_scatter_table = dict()

        for f in smhm_files:
            this_a = find_a(f)

            # Then use astropy to make tables out of those files
            smhm_file = self.smhm_dir + "smhm_a{}.dat".format(this_a)
            smhm_scatter_file = self.smhm_dir + "smhm_scatter_a{}.dat".format(this_a)

            self.smhm_table[this_a] = table.Table.read(smhm_file,
                                                       format="ascii")
            self.smhm_scatter_table[this_a] = table.Table.read(smhm_scatter_file,
                                                               format="ascii")

    def get_smhm(self, z, galaxy_type):
        """
        Get the stellar mass halo mass ratio and its intrinsic scatter at a
        given redshift

        :param z: Redshift at which to get the SMHM ratio
        :param galaxy_type: What kind of galaxies to select. Can be
                            "All", "Cen", "Cen_SF", "Cen_Q", "Sat", "SF", "Q".
        :return: A tuple of things. First is the redshift at which the SMHM
                 relation was accessed. Next are the halo masses at which the
                 SMHM relation is tabulated. The SM/HM ratio is next, with one
                 item for each mass. Next are the upper and lower boundaries of
                 the error region, which corresponds to the intrindic scatter
                 in the SMHM relation.
        """
        a = z_to_a(z)
        # the find closest function expects a sorted list
        a_smhm = find_closest(a, sorted(list(self.smhm_table.keys())))
        z_smhm = a_to_z(float(a_smhm))

        colnames = {"All": "True_Med_All(22)",
                    "Cen": "True_Cen(25)",
                    "Cen_SF": "True_Cen_SF(28)",
                    "Cen_Q": "True_Cen_Q(31)",
                    "Sat": "True_Sat(34)",
                    "SF": "True_Sat(37)",
                    "Q": "True_Q(40)"}

        colname = colnames[galaxy_type]

        smhm_log = self.smhm_table[a_smhm][colname].data
        # restrict to halo masses where the SMHM can be determined
        good_idxs = np.where(smhm_log < 0)
        smhm_log = smhm_log[good_idxs]
        smhm = 10 ** smhm_log

        halo_masses = 10 ** self.smhm_table[a_smhm]["HM(0)"].data[good_idxs]
        scatter_log = self.smhm_scatter_table[a_smhm][colname].data[good_idxs]

        # parse the scatter to be in linear space
        hi_lim = 10**(smhm_log + scatter_log)
        lo_lim = 10**(smhm_log - scatter_log)

        return z_smhm, halo_masses, smhm, hi_lim, lo_lim

    def read_sfh(self):
        # We will read in both the stellar mass and halo mass matched catalogs
        sfh_hm_files = [f for f in os.listdir(self.sfh_dir)
                        if f.startswith("sfh_hm")]
        sfh_sm_files = [f for f in os.listdir(self.sfh_dir)
                        if f.startswith("sfh_sm")]
        all_files = sfh_hm_files + sfh_sm_files

        # create dictionaries to store the eventual tables
        self.sfhs_hm = collections.defaultdict(dict)
        self.sfhs_sm = collections.defaultdict(dict)

        for f in all_files:
            this_a = find_a(f)
            this_m = find_m(f)

            if "sfh_sm" in f:
                result_dict = self.sfhs_sm
            elif "sfh_hm" in f:
                result_dict = self.sfhs_hm
            else:
                raise ValueError

            # skip reading since it's really computationally expensive with so
            # many files. for now just set things as None
            result_dict[this_a][this_m] = None

    def _read_individual_sfh(self, prefix, a, m):
        # Do the actual reading of the tables as needed, since this is
        # computationally expensive

        # strip off that extra zero we added earlier
        m = m.lstrip("0")
        # then load the correct file
        sfh_file = self.sfh_dir + prefix + "{}_a{}.dat".format(m, a)
        new_table = table.Table.read(sfh_file, format="ascii")
        # add a redshift column. Columns aren't named well, unfortunately
        new_table["z"] = a_to_z(new_table["col1"])

        # then put it where it belongs
        if prefix == "sfh_sm":
            self.sfhs_sm[a][m] = new_table
        elif prefix == "sfh_hm":
            self.sfhs_hm[a][m] = new_table


    def get_sfh(self, match, z, m):
        # Get the star formation history of a halo with mass m at redshift z
        # the match keyword tells whether this value is stellar mass or halo
        # mass
        if match == "halo":
            prefix = "sfh_hm"
            sfh_dict = self.sfhs_hm
        elif match == "stellar":
            prefix = "sfh_sm"
            sfh_dict = self.sfhs_sm
        else:
            raise ValueError

        a = z_to_a(z)
        # the find closest function expects a sorted list
        a_sfh = find_closest(a, sorted(list(sfh_dict.keys())))
        m_sfh = find_closest(np.log10(m), sorted(list(sfh_dict[a_sfh].keys())))

        # get the appropriate table.
        this_table = sfh_dict[a_sfh][m_sfh]
        # we didn't initialize the tables in read-in, since there are so many of
        # them it takes a long time. If we haven't already read in the table,
        # do so now
        if this_table is None:
            self._read_individual_sfh(prefix, a_sfh, m_sfh)
            this_table = sfh_dict[a_sfh][m_sfh]

        # we want to return redshift, sfh, error boundaries
        sfhs = this_table["col2"].data
        # only keep the ones where there is data
        good_idxs = np.where(sfhs > 0)
        sfhs = sfhs[good_idxs]

        redshifts = this_table["z"].data[good_idxs]
        sfh_err_hi = this_table["col3"].data[good_idxs]
        sfh_err_lo = this_table["col4"].data[good_idxs]

        hi_boundary = sfhs + sfh_err_hi
        lo_boundary = sfhs - sfh_err_lo
        return redshifts, sfhs, hi_boundary, lo_boundary


