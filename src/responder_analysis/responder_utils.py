import scipy
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.pylab import plt
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.proportion import proportion_confint


def z_test(p1: float, p2: float,
           n1: int, n2: int):
    """
    Test for proportions based on normal (z) test. (two-sided)

    Parameters
    ----------
    p1 : float
        proportion of successes in nobs trial (group 1).
    p2 : float
        proportion of successes in nobs trial (group 2).
    n1 : int
        total number of trails or observations (group 1).
    n2 : int
        total number og trails or observations (group 2).
    Returns
    -------
    p_value: float
        p_value from test for proportions based on normal z test.
    """
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * ((1 / n1) + (1 / n2)))
    z = (p1 - p2) / se
    p_value = scipy.stats.norm.sf(abs(z)) * 2
    return p_value


def binomial_confidence_interval(daily_response_df: pd.DataFrame,
                                 group_1_names: list,
                                 group_2_names: list,
                                 no_of_days: int,
                                 response_direction: str,
                                 success_percent: float,
                                 significance_level: float):
    """ Binomial confidence interval
    Produces the binomial confidence interval for two groups.

    Parameters
    ----------
    daily_response_df: pd.DataFrame
        Dataframe with percentage change from baseline .
    group_1_names: list
        List of names of patients in group 1.
    group_2_names: list
        List of names of patients in group 1.
    no_of_days: int
        Number of days of data to consider for the interval.
    response_direction: str
        Direction of change from baseline.
    success_percent: float
        Threshold for success criteria.
    significance_level: float
        Significance level

    Returns
    -------
    group_1_cnf_int, group_2_cnf_int: list
        group 1 and group 2 confidence interval tuples.
    """
    success_count_group_1 = None
    total_number_of_trial_group_1 = None
    success_count_group_2 = None
    total_number_of_trial_group_2 = None
    group_1_flat = pd.DataFrame(daily_response_df[group_1_names] \
                                .iloc[:no_of_days] \
                                .values.flatten()).dropna()
    group_2_flat = pd.DataFrame(daily_response_df[group_2_names] \
                                .iloc[:no_of_days] \
                                .values.flatten()).dropna()
    if response_direction == 'increase':
        success_count_group_1 = \
            group_1_flat[group_1_flat[0] > success_percent].shape[0]
        total_number_of_trial_group_1 = group_1_flat.shape[0]

        success_count_group_2 = \
            group_2_flat[group_2_flat[0] > success_percent].shape[0]
        total_number_of_trial_group_2 = group_2_flat.shape[0]
    elif response_direction == 'decrease':
        success_count_group_1 = \
            group_1_flat[group_1_flat[0] < -success_percent].shape[0]
        total_number_of_trial_group_1 = group_1_flat.shape[0]

        success_count_group_2 = \
            group_2_flat[group_2_flat[0] < -success_percent].shape[0]
        total_number_of_trial_group_2 = group_2_flat.shape[0]
    if success_count_group_1 is not None \
            or success_count_group_2 is not None \
            or total_number_of_trial_group_1 is not None \
            or total_number_of_trial_group_2 is not None:
        group_1_cnf_int = proportion_confint(success_count_group_1,
                                             total_number_of_trial_group_1,
                                             alpha=significance_level,
                                             method='normal')
        group_2_cnf_int = proportion_confint(success_count_group_2,
                                             total_number_of_trial_group_2,
                                             alpha=significance_level,
                                             method='normal')
        return [group_1_cnf_int, group_2_cnf_int]


def proportion_per_patient(response_percent_df: pd.DataFrame,
                           feature_name: str,
                           response_direction: str,
                           success_percent: float,
                           study_day: str,
                           group_1_names: list,
                           group_2_names: list,
                           no_of_days: int):
    """ Proportion of successful days per patient
        This function returns proportion of successful days per patient.

    Parameters
    ----------
    response_percent_df: pd.DataFrame
        Percentage response from baseline dataframe.
    feature_name: str
        Feature column name in response_percent_df in which proportion of
        successful days is calculated.
    response_direction: str
        Direction of response to be calculated. Ex: 'increase', 'decrease'.
    success_percent: float
        Success threshold for response.
    study_day: str
        Study day column name to be used.
    group_1_names: list
        list of group 1 names.
    group_2_names: list
        list of group 2 names.
    no_of_days: int
        number of days of data per patient.

    Returns
    -------
    tuple: group 1 and group 2 means.

    """
    response_percent_df_copy = response_percent_df.copy()

    if response_direction == 'increase':
        response_percent_df_copy['daily_success'] = response_percent_df_copy[
            feature_name] \
            .apply(lambda x_var: 1 if x_var > success_percent else 0)
    elif response_direction == 'decrease':
        response_percent_df_copy['daily_success'] = response_percent_df_copy[
            feature_name] \
            .apply(lambda x_var: 1 if x_var < -success_percent else 0)

    response_weeks = response_percent_df_copy[
        response_percent_df_copy[study_day] < no_of_days + 1]

    daily_success_matrix = response_weeks.pivot(index=study_day,
                                                columns='external_patient_id',
                                                values='daily_success')
    group_1_means = []
    for x in daily_success_matrix[group_1_names].columns:
        dict_val_count = dict(pd.DataFrame(
            daily_success_matrix[x][:no_of_days].dropna().values)
                              [0].value_counts(normalize=True))
        if dict_val_count.get(1) is not None:
            group_1_means.append(dict_val_count[1])
        else:
            group_1_means.append(0)

    group_2_means = []
    for x in daily_success_matrix[group_2_names].columns:
        dict_val_count = dict(pd.DataFrame(
            daily_success_matrix[x][:no_of_days].dropna().values)
                              [0].value_counts(normalize=True))
        if dict_val_count.get(1) is not None:
            group_2_means.append(dict_val_count[1])
        else:
            group_2_means.append(0)
    return group_1_means, group_2_means


def randomization_test(group_1_means: list,
                       group_2_means: list,
                       no_of_trials: int,
                       significance_level: float,
                       return_result: bool):
    """ Randomization test
    This performs a randomization test on two groups and tests
    for difference between two groups.

    Difference is taken as (group_1 - group_2)

    Parameters
    ----------
    group_1_means: list
        list of group 1 means.
    group_2_means: list
        list of group 2 means.
    no_of_trials: int
        number of trials (randomization parameter).
    significance_level: float
        significance level for hypothesis test.
    return_result: bool
        boolean parameter to return p value

    Returns
    -------
    p_value: float
        returns p-value for randomization test.

    """
    d = np.zeros((no_of_trials,))

    group_1_len = len(group_1_means)
    group_2_len = len(group_2_means)
    all_subjects = group_1_means + group_2_means

    diff_original = np.mean(group_1_means) - \
                    np.mean(group_2_means)

    label_arr = list(np.ones(group_1_len).astype(int)) + \
                list(np.zeros(group_2_len).astype(int))
    for x in range(no_of_trials):
        np.random.shuffle(label_arr)
        proportion_df = pd.DataFrame({'subjects_means': all_subjects,
                                      'label': label_arr})
        group_1_x = proportion_df.groupby(['label']).mean().loc[1].values[0]
        group_2_x = proportion_df.groupby(['label']).mean().loc[0].values[0]
        d[x] = group_1_x - group_2_x

    p = len(np.where(d >= diff_original)[0])
    p_value = p / float(no_of_trials)
    if return_result:
        return p_value
    print('\033[1m' + "\nRandomization test results:\n" + "\033[0;0m")
    print("Original difference between group 1 and group 2:", diff_original)
    print("p-value:", p_value)
    if p_value < significance_level:
        print(
            "Result: randomization test suggests significant difference "
            "between two groups.")
    else:
        print(
            "Result: randomization test suggests no significant difference "
            "between two groups.")

    plt.figure(figsize=(8, 5))
    plt.hist(d)
    plt.axvline(x=diff_original)
    plt.title("Distribution of Group Differences")
    plt.xlabel("Mean Difference")
    plt.ylabel("Frequency")


def weekly_confidence_interval_plot(daily_response_matrix: pd.DataFrame,
                                    weekly_response_matrix: pd.DataFrame,
                                    group_1_names: list, group_2_names: list,
                                    response_direction: str,
                                    success_percent: float,
                                    significance_level: float,
                                    no_of_weeks: int,
                                    group_1_color: str, group_2_color: str,
                                    group_one_name: str, group_two_name: str):
    """ Weekly confidence interval plot
    Creates a weekly confidence interval plot visualization

    Parameters
    ----------
    daily_response_matrix: pd.DataFrame
        Daily response matrix
    weekly_response_matrix: pd.DataFrame
        Weekly response matrix
    group_1_names: list
        List of group 1 names
    group_2_names: list
        List of group 2 names
    response_direction: str
        Response direction (e.g: 'increase' or 'decrease')
    success_percent: float
        Success threshold for response
    significance_level: float
        Significance level for hypothesis testing
    no_of_weeks: int
        Number of weeks of treatment period
    group_1_color: str
        Group 1 color
    group_2_color: str
        Group 2 color
    group_one_name: str
        Group 1 name
    group_two_name: str
        Group 2 name

    Returns
    -------

    """
    success_count_group_1 = None
    total_number_of_trial_group_1 = None
    success_count_group_2 = None
    total_number_of_trial_group_2 = None
    group_1_weekly_cnf_int = []
    group_2_weekly_cnf_int = []
    for i, week in daily_response_matrix.groupby(
            np.arange(len(daily_response_matrix)) // 7):
        weekly_group_1 = pd.DataFrame(week[group_1_names].values.flatten()) \
            .dropna().values.flatten()
        weekly_group_2 = pd.DataFrame(week[group_2_names].values.flatten()) \
            .dropna().values.flatten()
        if response_direction == 'increase':
            success_count_group_1 = \
                weekly_group_1[weekly_group_1 > success_percent].shape[0]
            total_number_of_trial_group_1 = weekly_group_1.shape[0]
            success_count_group_2 = \
                weekly_group_2[weekly_group_2 > success_percent].shape[0]
            total_number_of_trial_group_2 = weekly_group_2.shape[0]
        elif response_direction == 'decrease':
            success_count_group_1 = \
                weekly_group_1[weekly_group_1 < -success_percent].shape[0]
            total_number_of_trial_group_1 = weekly_group_1.shape[0]
            success_count_group_2 = \
                weekly_group_2[weekly_group_2 < -success_percent].shape[0]
            total_number_of_trial_group_2 = weekly_group_2.shape[0]

        group_1_cnf_int = proportion_confint(success_count_group_1,
                                             total_number_of_trial_group_1,
                                             alpha=significance_level,
                                             method='normal')
        group_2_cnf_int = proportion_confint(success_count_group_2,
                                             total_number_of_trial_group_2,
                                             alpha=significance_level,
                                             method='normal')
        group_1_weekly_cnf_int.append(group_1_cnf_int)
        group_2_weekly_cnf_int.append(group_2_cnf_int)
        if i == no_of_weeks - 1:
            break

    plt.figure(figsize=(10, 5))

    x_ind = weekly_response_matrix.index

    group_1_lower = pd.DataFrame(group_1_weekly_cnf_int)[0].values
    group_1_upper = pd.DataFrame(group_1_weekly_cnf_int)[1].values

    group_2_lower = pd.DataFrame(group_2_weekly_cnf_int)[0].values
    group_2_upper = pd.DataFrame(group_2_weekly_cnf_int)[1].values

    hor_width = 0.08

    for x_i, group_1_l, group_1_u in zip(x_ind,
                                         group_1_lower,
                                         group_1_upper):
        plt.axvline(x=x_i, ymin=group_1_l,
                    ymax=group_1_u,
                    color=group_1_color)
        plt.hlines(y=group_1_l,
                   xmin=x_i - hor_width,
                   xmax=x_i + hor_width,
                   color=group_1_color)
        plt.hlines(y=group_1_u,
                   xmin=x_i - hor_width,
                   xmax=x_i + hor_width,
                   color=group_1_color)

    for x_i, group_2_l, group_2_u in zip(x_ind,
                                         group_2_lower,
                                         group_2_upper):
        plt.axvline(x=x_i, ymin=group_2_l,
                    ymax=group_2_u,
                    color=group_2_color)
        plt.hlines(y=group_2_l,
                   xmin=x_i - hor_width,
                   xmax=x_i + hor_width,
                   color=group_2_color)
        plt.hlines(y=group_2_u,
                   xmin=x_i - hor_width,
                   xmax=x_i + hor_width,
                   color=group_2_color)

    weekly_response_matrix[group_1_names] \
        .iloc[:no_of_weeks] \
        .mean(axis=1) \
        .plot(label=group_one_name,
              color=group_1_color)

    weekly_response_matrix[group_2_names] \
        .iloc[:no_of_weeks] \
        .mean(axis=1) \
        .plot(label=group_two_name,
              color=group_2_color)

    plt.title('Weekly proportion of successful days (treatment effect)')
    plt.xlabel('Week name')
    plt.ylabel('Treatment effects')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def t_test(group_1_means: list, group_2_means: list,
           sig_level: float,
           display: bool):
    """

    Parameters
    ----------
    group_1_means: list
        Group 1 means
    group_2_means: list
        Group 2 means
    sig_level: float
        Significance level for hypothesis testing
    display: bool
        Boolean variable whether to display ttest results.

    Returns
    -------

    """
    # 1) Equality of variance test
    equality_of_variance_test = stats.levene(group_1_means, group_2_means)
    # 2) Normality test
    normality_test = (
        stats.shapiro(group_1_means), stats.shapiro(group_2_means))

    if (equality_of_variance_test[1] > sig_level) & \
            (normality_test[0][1] > sig_level) & \
            (normality_test[1][1] > sig_level):
        if display:
            flag_equal_var = {True: 'Success', False: 'Fails'}

            equality_of_var_pvalue = "{:.4f}".format(
                equality_of_variance_test[1])
            shapiro_pvalue_group_1 = "{:.4f}".format(normality_test[0][1])
            shapiro_pvalue_group_2 = "{:.4f}".format(normality_test[1][1])

            equal_var_result = flag_equal_var[
                float(equality_of_var_pvalue) > sig_level]
            shapiro_result = flag_equal_var[
                (float(shapiro_pvalue_group_1) > sig_level) &
                (float(shapiro_pvalue_group_2) > sig_level)]

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
                float(equality_of_var_pvalue) > sig_level]
            shapiro_result = flag_equal_var[
                (float(shapiro_pvalue_group_1) > sig_level) &
                (float(shapiro_pvalue_group_2) > sig_level)]

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
