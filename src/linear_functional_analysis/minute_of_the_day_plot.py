import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from matplotlib.pylab import plt

import seaborn as sns

palette = sns.color_palette('deep', 10)
plt.style.use('seaborn-whitegrid')
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)


def smooth_activity_pattern(activity_df: pd.DataFrame,
                            patient_id: str, color: str,
                            time_period: str, degree_poly: int):
    """
    This function is used to visualize minute of the day smooth activity pattern
    per patient.
    Parameters
    ----------
    activity_df: pd.DataFrame
        High resolution wearables activity data with activity label. This data
        is expected at second-level.
    patient_id: str
        Patient id on which smooth activity pattern is to be visualized.
    color: str
        Color of the study group which is used to name the title of the plot.
    time_period: str
        Time period of the study to visualize. For e.g. "Baseline", "Treatment".
    degree_poly: str
        Degree of polynomial features (smooth curve).

    Returns
    -------

    """
    data_activity = activity_df.copy()
    data_activity['scratching'] = 1
    data_activity['minute'] = data_activity['timestamp'].map(lambda x: x.minute)
    data_activity['hour'] = data_activity['timestamp'].map(lambda x: x.hour)
    data_activity['minute_of_day'] = data_activity['hour'] * 60 + \
                                     data_activity['minute']
    minute_of_day_scatter_groupby = data_activity.groupby(['minute_of_day']) \
        .count()[['scratching']] \
        .reindex(range(1440)).fillna(0)

    X = pd.DataFrame(np.array(range(1440)))
    y = minute_of_day_scatter_groupby.values
    model = make_pipeline(PolynomialFeatures(degree_poly), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X)
    scaler = MinMaxScaler()
    data_sample_baseline = pd.DataFrame(scaler.fit_transform(y_plot \
                                                             .reshape(-1, 1)))
    plt.figure(figsize=(10, 5))
    plt.title("Patient:" + patient_id + '| Time period:' + time_period + \
              " | Color:" + color,
              color='Black', fontdict={'fontsize': 15})
    plt.plot(data_sample_baseline.index, data_sample_baseline.values,
             color=palette[1])
    plt.xlabel('Scratching baseline Hours', fontsize=10)


def minute_of_the_day_smooth_activity_pattern(activity_df: pd.DataFrame,
                                              patient_id: str,
                                              color: str,
                                              time_period: str,
                                              activity_name: str,
                                              range_degree: list):
    """
    This helper function is used to visualize minute of the day smooth
    activity pattern along with original data points scattered.
    Parameters
    ----------
    activity_df: pd.DataFrame
        High resolution wearables activity dataframe with activity label. This
        is seconds-level activity data.
    patient_id: str
        Patient id on which smooth activity pattern is to be visualized.
    color: str
        Color of the study group which is used to name the title of the plot.
    time_period: str
        Time period of the study to visualize. For e.g. "Baseline", "Treatment".
    activity_name: str
        Activity label to visualize (E.g. "Scratching", "Walking", "Running",
        "Resting", "Sleeping", "Shaking")
    range_degree: str
        Range of degree of polynomial features. This is used to plot multiple
        degree curves.

    Returns
    -------

    """
    data_activity = activity_df.copy()
    data_activity['scratching'] = 1
    data_activity['minute'] = data_activity['timestamp'].map(lambda x: x.minute)
    data_activity['hour'] = data_activity['timestamp'].map(lambda x: x.hour)
    data_activity['minute_of_day'] = data_activity['hour'] * 60 + \
                                     data_activity['minute']
    minute_of_day_scatter_groupby = data_activity.groupby(['minute_of_day']) \
        .count() \
        [['scratching']].reindex(range(1440)) \
        .fillna(0)
    X = pd.DataFrame(np.array(range(1440)))
    y = minute_of_day_scatter_groupby.values
    plt.figure(figsize=(20, 8))
    plt.title(patient_id + '| Time period:' + time_period + '| Color:' + \
              color, fontsize=16)
    plt.scatter(minute_of_day_scatter_groupby.index,
                minute_of_day_scatter_groupby.values, c='black',
                label='Ground truth Scratching/min')
    locs, labs = plt.xticks()
    plt.xticks(locs,
               [str(x) + ' Hour ' + str(y) + ' Min' for x, y in zip((locs /
                                                                     60) \
                   .astype(
                   int), (locs % 60).astype(int))])
    X = pd.DataFrame(np.array(range(1440)))
    y = minute_of_day_scatter_groupby.values
    for count, degree in enumerate(range_degree):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        y_plot = model.predict(X)
        plt.plot(y_plot, label="degree %d" % degree)
    plt.xlabel('Minutes 0-1440')
    plt.ylabel(activity_name + ' activity at Minute(0-1440) Total')
    plt.legend(loc='upper right')

    plt.show()
