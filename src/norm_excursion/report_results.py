import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.pylab import plt
from statsmodels.iolib.table import SimpleTable


class ReportResults:
    """ Report Results

    Report results class is used to present all the key results from norm
    excursion analyses. It has helper functions to visualize treatment effects,
    report t-test statistics, visualizing histogram of proportion of
    successful days per group, reporting confidence intervals, running
    difference in proportion test (binomial test) and Mann whitney test.

    Parameters
    ----------
    daily_patient_analysis : pd.DataFrame
        Daily treatment effects for all patients using norm excursion (in
        treatment period).
    weekly_patient_analysis : pd.DataFrame
        Weekly treatment effects for all patients using norm excursion (in
        treatment period).
    group_1_names : list
        List of group 1 patient names.
        For e.g. ['CASE-SAMPLE-01','CASE-SAMPLE-02']
    group_2_names : list
        List of group 2 patient names.
        For e.g. ['CASE-SAMPLE-03','CASE-SAMPLE-04']
    group_1 : str
        Group 1 study group name.
    group_2 : str
        Group 2 study group name.
    color_1 : tuple
        Group 1 color code (RGB)
        For e.g. (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    color_2 : tuple
        Group 2 color code (RGB)
        For e.g. (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    number_of_days : int
        Number of days of data to filter in treatment period for each patient
        when running t-test.
    sig_level : float
        Significance level used in T-test.
  """

    def __init__(self, daily_patient_analysis: pd.DataFrame,
                 weekly_patient_analysis: pd.DataFrame,
                 group_1_names: list, group_2_names: list,
                 group_1: str, group_2: str,
                 color_1: tuple, color_2: tuple,
                 number_of_days: int,
                 sig_level: float):
        self.daily_patient_analysis = daily_patient_analysis
        self.weekly_patient_analysis = weekly_patient_analysis
        self.group_1_names = group_1_names
        self.group_2_names = group_2_names
        self.group_1 = group_1
        self.group_2 = group_2
        self.color_1 = color_1
        self.color_2 = color_2
        self.number_of_days = number_of_days
        self.sig_level = sig_level
        self.means_of_groups = self.report_ttest_statistic(False)

    def visualize_treatment_effect(self, daily_flag: bool,
                                   time_period_daily: int,
                                   no_of_weeks: int):
        """ Visualizing daily/weekly treatment effects per group

        This function is used to visualize daily/weekly treatment effects per
        group to understand if there is significant treatment effects in a
        specific group.

        Parameters
        ----------
        daily_flag : bool
            Set this to "True" to visualize daily treatment effects else set
            this to "False" to visualize weekly treatment effects.
        time_period_daily: int
            Set this parameter to the number of days you want to visualize when
            using daily visualization.
        no_of_weeks: int
            Set this parameter to the number of weeks you want to visualize when
            using weekly visualization.

        """
        if daily_flag:
            daily_success_df = pd.concat(self.daily_patient_analysis).pivot(
                index='day_name', columns='external_patient_id',
                values='success_days')
            plt.figure(figsize=(20, 5))
            daily_success_df[self.group_1_names] \
                .mean(axis=1)[:time_period_daily] \
                .plot(label=self.group_1, color=self.color_1)
            daily_success_df[self.group_2_names] \
                .mean(axis=1)[:time_period_daily] \
                .plot(label=self.group_2, color=self.color_2)
            plt.ylim(0, 1)
            plt.ylabel('Treatment effects')
            plt.title("Daily treatment effects (proportion of patients having norm excursion) visualization per group")
            plt.legend()

        else:
            weekly_success_df = pd.concat(self.weekly_patient_analysis).pivot(
                index='week_name', columns='external_patient_id',
                values='percent_of_success_days')
            plt.figure(figsize=(10, 5))
            weekly_success_df[self.group_1_names].iloc[:no_of_weeks] \
                .mean(axis=1).plot(label=self.group_1, color=self.color_1)
            weekly_success_df[self.group_2_names].iloc[:no_of_weeks] \
                .mean(axis=1).plot(label=self.group_2, color=self.color_2)
            plt.ylim(0, 1)
            plt.ylabel('Treatment effects')
            plt.title("Weekly treatment effects visualization per group")
            plt.legend()

    def report_ttest_statistic(self, display: bool):
        """ Report t-test statistic

        This function is used to report t-test statistic and also check if
        t-test assumptions are met.

        Parameters
        ----------
        display: bool
            Setting display to False returns group 1 means and group 2 means.
            Setting display to True prints t-test results in tabular format.

        Returns
        -------
        :rtype: (list, list)
        group_1_means: list
            Proportion of successful days in group 1 patients.
        group_2_means: list
            Proportion of successful days in group 2 patients.

        """
        daily_success_df = pd.concat(self.daily_patient_analysis) \
            .pivot(index='day_name',
                   columns='external_patient_id',
                   values='success_days')
        group_1_means = []
        for x in daily_success_df[self.group_1_names].columns:
            # Dropping NULL values after filtering the dataframe using number of
            # days of data.
            dict_val_count = dict(pd.DataFrame(
                daily_success_df[x][:self.number_of_days].dropna().values)
                                  [0].value_counts(normalize=True))
            if dict_val_count.get(1) is not None:
                group_1_means.append(dict_val_count[1])
            else:
                group_1_means.append(0)

        group_2_means = []
        for x in daily_success_df[self.group_2_names].columns:
            dict_val_count = dict(pd.DataFrame(
                daily_success_df[x][:self.number_of_days].dropna().values)
                                  [0].value_counts(normalize=True))
            if dict_val_count.get(1) is not None:
                group_2_means.append(dict_val_count[1])
            else:
                group_2_means.append(0)
        # 1) Equality of variance test
        equality_of_variance_test = stats.levene(group_1_means, group_2_means)
        # 2) Normality test
        normality_test = (
            stats.shapiro(group_1_means), stats.shapiro(group_2_means))

        if (equality_of_variance_test[1] > self.sig_level) & \
                (normality_test[0][1] > self.sig_level) & \
                (normality_test[1][1] > self.sig_level):
            if display:
                flag_equal_var = {True: 'Success', False: 'Fails'}

                equality_of_var_pvalue = "{:.4f}".format(
                    equality_of_variance_test[1])
                shapiro_pvalue_group_1 = "{:.4f}".format(normality_test[0][1])
                shapiro_pvalue_group_2 = "{:.4f}".format(normality_test[1][1])

                equal_var_result = flag_equal_var[
                    float(equality_of_var_pvalue) > self.sig_level]
                shapiro_result = flag_equal_var[
                    (float(shapiro_pvalue_group_1) > self.sig_level) &
                    (float(shapiro_pvalue_group_2) > self.sig_level)]

                mydata = [[equal_var_result, equality_of_var_pvalue],
                          [shapiro_result, ''],
                          ['', shapiro_pvalue_group_1],
                          ['', shapiro_pvalue_group_2]]
                myheaders = ["Success/Fails", 'P-value']
                mystubs = ["1) Test for equality of variance (levene)",
                           "2) Test for normality (shapiro)",
                           " a) Group 1",
                           " b) Group 2"]
                tbl = SimpleTable(mydata, myheaders, mystubs,
                                  title="T-test Assumptions")
                print(tbl)
                print("Levene test")
                print(
                    "The Levene test tests the null hypothesis that all input "
                    "\nsamples are from populations with equal variances.")
                print('-' * 63)
                print("Shapiro test")
                print(
                    "The Shapiro-Wilk test tests the null hypothesis that the "
                    "\ndata was drawn from a normal distribution.")
                print('-' * 63)
                print("\n\n")
                t_test_result = stats.ttest_ind(group_1_means, group_2_means)
                t_test_flag = {True: 'Reject the null hypothesis',
                               False: 'Do not reject the null hypothesis'}
                t_test_conclusion = t_test_flag[t_test_result[1] < 0.05]
                mydata_ttest = [
                    [str(t_test_result[0])[:6], str(t_test_result[1])[:6],
                     t_test_conclusion]]
                myheaders_ttest = ["Statistic", 'P-value', 'Conclusion']
                mystubs_ttest = ["Results"]
                tbl_ttest = SimpleTable(mydata_ttest, myheaders_ttest,
                                        mystubs_ttest,
                                        title="T-test (difference in "
                                              "proportion of outlier)")
                print(tbl_ttest)
            else:
                return group_1_means, group_2_means
        else:
            if display:
                flag_equal_var = {True: 'Success', False: 'Fails'}

                equality_of_var_pvalue = "{:.4f}".format(
                    equality_of_variance_test[1])
                shapiro_pvalue_group_1 = "{:.4f}".format(normality_test[0][1])
                shapiro_pvalue_group_2 = "{:.4f}".format(normality_test[1][1])

                equal_var_result = flag_equal_var[
                    float(equality_of_var_pvalue) > self.sig_level]
                shapiro_result = flag_equal_var[
                    (float(shapiro_pvalue_group_1) > self.sig_level) &
                    (float(shapiro_pvalue_group_2) > self.sig_level)]

                mydata = [[equal_var_result, equality_of_var_pvalue],
                          [shapiro_result, ''],
                          ['', shapiro_pvalue_group_1],
                          ['', shapiro_pvalue_group_2]]
                myheaders = ["Success/Fails", 'P-value']
                mystubs = ["1) Test for equality of variance (levene)",
                           "2) Test for normality (shapiro)",
                           " a) Group 1",
                           " b) Group 2"]
                tbl = SimpleTable(mydata, myheaders, mystubs,
                                  title="T-test Assumptions")
                print(tbl)
                print("Levene test")
                print(
                    "The Levene test tests the null hypothesis that all input "
                    "\nsamples are from populations with equal variances.")
                print('-' * 63)
                print("Shapiro test")
                print(
                    "The Shapiro-Wilk test tests the null hypothesis that the "
                    "\ndata was drawn from a normal distribution.")
                print('-' * 63)
                print("\n\n")
                t_test_result = stats.ttest_ind(group_1_means, group_2_means)
                t_test_flag = {True: 'Reject the null hypothesis',
                               False: 'Do not reject the null hypothesis'}
                t_test_conclusion = t_test_flag[t_test_result[1] < 0.05]
                mydata_ttest = [
                    [str(t_test_result[0])[:6], str(t_test_result[1])[:6],
                     t_test_conclusion]]
                myheaders_ttest = ["Statistic", 'P-value', 'Conclusion']
                mystubs_ttest = ["Results"]
                tbl_ttest = SimpleTable(mydata_ttest, myheaders_ttest,
                                        mystubs_ttest,
                                        title="T-test (difference in "
                                              "proportion of outlier)")
                print(tbl_ttest)
            else:
                return group_1_means, group_2_means

    def plot_fraction_of_outlier_hist(self, kde_binwidth: float = 0.1):
        """ Plot fraction of outlier histogram
        This function is used to plot the fraction of outliers per patient and
        per group.

        kde_binwidth: float
            Width of each bin in the histogram plot.
        """
        plt.figure(figsize=(10, 5))
        plt.title("Histogram of means of group 1 and group 2 (treatment "
                  "effects/outliers)")
        sns.histplot(self.means_of_groups[0], color=self.color_1,
                     label=self.group_1, kde='True', binwidth=kde_binwidth)
        sns.histplot(self.means_of_groups[1], color=self.color_2,
                     label=self.group_2, kde='True', binwidth=kde_binwidth)

    @staticmethod
    def confidence_interval(p: float, n: int):
        """ Confidence interval

        This static function is used to report binomial proportion confidence
        interval.

        Parameters
        ----------
        p: float
            Proportion of outlying/successful days
        n: int
            Count of the number of values used to calculate p.

        Returns
        -------
        :rtype: (list, list)
            Returns the range of confidence interval as tuple. For Example:
            (0.62, 0.68)

        References
        ----------
        https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        """
        # For a 95% confidence level, the error alpha = 1 - 0.95 = 0.05, so
        # 1 - (alpha/2) = 0.975 and z_alpha_by_2 = 1.96.
        # Reference link included above for z_alpha_by_2 value.
        z_alpha_by_2 = 1.96
        return ((p - (z_alpha_by_2 * np.sqrt((p * (1 - p)) / n))),
                (p + (z_alpha_by_2 * np.sqrt((p * (1 - p)) / n))))

    def report_confidence_interval(self):
        """ Report Confidence Interval
        This function is used to report binomial proportion confidence interval
        for group 1 and group 2.
        """
        daily_success_df = pd.concat(self.daily_patient_analysis) \
            .pivot(index='day_name',
                   columns='external_patient_id',
                   values='success_days')
        group_1_tuple = \
            pd.DataFrame(daily_success_df[self.group_1_names].values.flatten(

            )).dropna()[0].value_counts(normalize=True)[1], \
            pd.DataFrame(daily_success_df[self.group_1_names].values
                         .flatten()).dropna().shape[0]
        group_2_tuple = \
            pd.DataFrame(daily_success_df[self.group_2_names].values.flatten(

            )).dropna()[0].value_counts(normalize=True)[1], \
            pd.DataFrame(daily_success_df[self.group_2_names].values
                         .flatten()).dropna().shape[0]
        print("Group 1 confidence interval (Based on fraction of outlier)",
              self.confidence_interval(group_1_tuple[0], group_1_tuple[1]))
        print("Group 2 confidence interval (Based on fraction of outlier)",
              self.confidence_interval(group_2_tuple[0], group_2_tuple[1]))

    @staticmethod
    def z_test(p1: float, p2: float, n1: int, n2: int, sig_level: float):
        """ Z test

        Binomial test for difference in proportion.

        Parameters
        ----------
        p1: float
            Proportion in group 1.
        p2: float
            Proportion in group 2.
        n1: int
            number of values in group 1.
        n2: int
            number of values in group 2.
        sig_level: float
            Significance level above which we cannot reject the null hypothesis.

        References
        ----------
        https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/z-test/
        """
        print("H0: P1=P2 (p1 is same as p2)")
        print("Ha: P1!=P2 (p1 is not same as p2)")
        print("\nResults:")
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * ((1 / n1) + (1 / n2)))
        z = (p1 - p2) / se
        p_value = scipy.stats.norm.sf(abs(z)) * 2
        print("P-value:", '{:.20f}'.format(p_value))
        if p_value < sig_level:
            print("Reject the null hypothesis that p1 is the same as p2.")
            print("There is significant difference")
        else:
            print(
                "Cannot Reject the null hypothesis that p1 is the same as p2.")

    def binomial_test(self, sig_level: float, return_params: bool):
        """ Binomial test

        This function is used to perform binomial test (difference in proportion
        test) and report it.

        Parameters
        ----------
        sig_level: float
            Significance level above which we cannot reject null hypothesis.
        return_params: bool
            returns parameters when true otherwise prints the results.
        """
        daily_success_df = pd.concat(self.daily_patient_analysis) \
            .pivot(index='day_name',
                   columns='external_patient_id',
                   values='success_days')
        group_1_1 = pd.DataFrame(daily_success_df[self.group_1_names]
                                 .values.flatten()).dropna()[0] \
            .value_counts()[1]
        group_1_0 = pd.DataFrame(daily_success_df[self.group_1_names]
                                 .values.flatten()).dropna()[0] \
            .value_counts()[0]
        p1 = group_1_1 / (group_1_1 + group_1_0)

        group_2_1 = pd.DataFrame(daily_success_df[self.group_2_names]
                                 .values.flatten()).dropna()[0] \
            .value_counts()[1]
        group_2_0 = pd.DataFrame(daily_success_df[self.group_2_names]
                                 .values.flatten()).dropna()[0] \
            .value_counts()[0]
        p2 = group_2_1 / (group_2_1 + group_2_0)

        n1 = group_1_1 + group_1_0
        n2 = group_2_1 + group_2_0
        if return_params:
            return {'p1': p1, 'p2': p2, 'n1': n1, 'n2': n2}
        else:
            self.z_test(p1, p2, n1, n2, sig_level)

    def mann_whitney_test(self):
        """ Mann-Whitney rank
        Compute the Mann-Whitney rank test on samples means group 1 and group 2.
        """
        return stats.mannwhitneyu(self.means_of_groups[0],
                                  self.means_of_groups[1])
