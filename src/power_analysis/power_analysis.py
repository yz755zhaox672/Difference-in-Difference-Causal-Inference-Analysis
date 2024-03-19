import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy.stats as st
from typing import Tuple
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower, TTestIndPower


class PowerAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def get_proportion_effectsize(x1: float, x2: float):
        """ Proportion effect size

        This helper function is used to calculate effect size using proportion
        of successful days per group

        Parameters
        ----------
        x1: float
            Group 1 proportion of successful days
        x2: float
            Group 2 proportion of successful days
        """
        return sm.stats.proportion_effectsize(x1, x2)

    @staticmethod
    def proportion_z_test(x1: float, x2: float):
        """ Proportion z test

        This helper function is used to perform z test on proportion of
        successful days per group

        Parameters
        ----------
        x1: float
            Group 1 proportion of successful days
        x2: float
            Group 2 proportion of successful days
        """
        return proportions_ztest(x1, x2)

    @staticmethod
    def get_ttestind(x1: np.array, x2: np.array):
        """ T-test

        This function is used to perform t-test on means of groups

        Parameters
        ----------
        x1: np.array
            Means of group 1
        x2: np.array
            Means of group 2
        """
        return st.ttest_ind(x1, x2)

    @staticmethod
    def man_whitney_test(x1: np.array, x2: np.array):
        """ Mann-Whitney rank
        Compute the Mann-Whitney rank test on samples group 1 and group 2.

        Parameters
        ----------
        x1: np.array
            Means of group 1
        x2: np.array
            Means of group 2
        """
        return stats.mannwhitneyu(x1, x2)

    @staticmethod
    def get_cohend_effectsize(d1: np.array, d2: np.array):
        """ Cohen d effect size

        This function is used to calculate effect size using cohen d method

        Parameters
        ----------
        d1: np.array
            Group 1 means
        d2: np.array
            Group 2 means
        """
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1), np.mean(d2)
        # calculate the effect size
        return (u1 - u2) / s

    @staticmethod
    def pooled_sd(d1: np.array, d2: np.array):
        """ Pooled standard deviation

        This function is used to calculate pooled standard deviation

        Parameters
        ----------
        d1: np.array
            Group 1 means
        d2: np.array
            Group 2 means
        """
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        return s

    @staticmethod
    def power_binomial(p1: float, p2: float):
        """ Used to calculate number of samples required per group using
        proportion of successful days

        Parameters
        ----------
        p1: float
            Group 1 proportion of successful days
        p2: float
            Group 2 proportion of successful days

        """
        # Based on critical values for 0.05 significance
        z_alpha = 1.96
        # Based on 80% power
        z_beta = 0.84
        n = (np.square(z_alpha + z_beta) * (
                p1 * (1 - p1) + (p2 * (1 - p2)))) / np.square(p1 - p2)
        return print("Total number of samples required per group: "
                     "(@ 0.80 power)", n)

    @staticmethod
    def power_means(se: float, d: float):
        """ Calculating number of samples required using means of groups

        Parameters
        ----------
        se: float
            Pooled standard deviation
        d: float
            difference of mean between group 1 and group 2
        """
        # Based on critical values for 0.05 significance
        z_alpha = 1.96
        # Based on 80% power
        z_beta = 0.84
        n = (2 * (np.square(z_alpha + z_beta)) * (np.square(se))) / (
            np.square(d))
        return print("Number of samples required per group:", n)

    def simulate_single_experiment(self, sample_size: int,
                                   group_1_means: np.array,
                                   group_2_means: np.array,
                                   sig_level: float):
        """ Simulating from means of groups from random normal

        Parameters
        ----------
        sample_size: int
            Sample size used to simulate
        group_1_means: np.array
            Group 1 means
        group_2_means: np.array
            Group 2 means
        sig_level: float
            significance level required
        """
        control_mean = np.mean(group_2_means)
        control_sd = np.std(group_2_means)
        treatment_mean = np.mean(group_1_means)
        treatment_sd = np.std(group_1_means)

        control_time_spent = np.random.normal(loc=control_mean,
                                              scale=control_sd,
                                              size=sample_size)
        treatment_time_spent = np.random.normal(loc=treatment_mean,
                                                scale=treatment_sd,
                                                size=sample_size)
        t_stat, p_value = st.ttest_ind(treatment_time_spent, control_time_spent)
        print("P-value: {}, Statistically Significant? {}"
              .format(p_value, p_value < sig_level))
        print("Effect size:", self.get_cohend_effectsize(treatment_time_spent,
                                                         control_time_spent))
        plt.hist(control_time_spent)
        plt.hist(treatment_time_spent)
        plt.show()

    @staticmethod
    def power_ttestind(effect_size: float, power_req: float,
                       nobs_range: Tuple[int], sig_level: float):
        """ Calculating sample size required using means of groups and plot
        power vs sample size.

        Parameters
        ----------
        effect_size: float
            effect size calculated from cohen d method
        power_req: float
            Power required
        nobs_range: tuple
            range of samples size to plot the power
        sig_level: float
            significance level
        """
        analysis = TTestIndPower()
        result = analysis.solve_power(effect_size, power=power_req, nobs1=None,
                                      ratio=1.0, alpha=sig_level,
                                      alternative='two-sided')
        print('Total Samples per group: %.3f' % int(result))

        TTestIndPower().plot_power(dep_var='nobs',
                                   nobs=np.arange(nobs_range[0],
                                                  nobs_range[1]),
                                   effect_size=[effect_size],
                                   alpha=sig_level,
                                   title='Power of t-Test' + '\n' +
                                         r'$\alpha = $' + str(sig_level))
        plt.axhline(y=power_req, color='r', linestyle='-')
        plt.axvline(x=result, color='g', linestyle='-')
        plt.text(result + 5, power_req + 0.02,
                 '(Power=' + str(power_req)
                 + ' Sample_size (Per group)= ' +
                 str(int(result)) + ')',
                 rotation=0)

    @staticmethod
    def power_simulations(n_range: int, num_sim: int,
                          group_1_means: np.array, group_2_means: np.array):
        """ Power simulations
        Simulating power analysis using data from norm excursion (Mean and SD)
        of treatment and control group. Then simply power is fraction of
        experiments where we reject the null hypothesis.

        Parameters
        ----------
        n_range: int
            Number of iterations of simulations
        num_sim: int
            Number of simulations to run per iteration
        group_1_means: np.array
            Group 1 means
        group_2_means: np.array
            Group 2 means
        """
        power_append = []
        for _ in range(n_range):
            sample_size = 10
            control_mean = np.array(group_2_means).mean()
            control_sd = np.array(group_2_means).std()
            treatment_mean = np.array(group_1_means).mean()
            treatment_sd = np.array(group_1_means).std()
            sims = num_sim
            while 1:
                control_time_spent = np.random.normal(loc=control_mean,
                                                      scale=control_sd,
                                                      size=(sample_size, sims))
                treatment_time_spent = np.random.normal(loc=treatment_mean,
                                                        scale=treatment_sd,
                                                        size=(
                                                            sample_size, sims))
                t, p = st.ttest_ind(treatment_time_spent, control_time_spent)
                power = (p < 0.05).sum() / sims
                if power >= 0.8:
                    break
                else:
                    sample_size += 1
            power_append.append(sample_size)
        print("Sample size required:", np.mean(power_append))

    @staticmethod
    def bernoulli_trial_simulation(p1, p2, sims):
        """ Bernoulli trial power simulation
        Simulating bernoulli trial and analyze the sample size required for
        power = 80%. Here the fraction of experiments rejecting the null
        hypothesis represents power.

        Parameters
        ----------
        p1: float
            Proportion of successful days in group 1
        p2: float
            Proportion of successful days in group 2
        sims: int
            Number of simulations to run
        """
        sample_size = 10
        while 1:
            control_time_spent = np.random.binomial(sample_size, p2, sims)
            treatment_time_spent = np.random.binomial(sample_size, p1, sims)
            p = []
            for x in range(0, sims):
                tval, pval = proportions_ztest(
                    [treatment_time_spent[x], control_time_spent[x]],
                    [sample_size, sample_size])
                p.append(pval)
            p = np.array(p)
            power = (p < 0.05).sum() / sims
            if power >= 0.8:
                break
            else:
                sample_size += 5
        print("For 80% power, sample size required = {}".format(sample_size))
