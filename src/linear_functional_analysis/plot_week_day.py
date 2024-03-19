from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.pylab import plt
import pandas as pd
import seaborn as sns

palette = sns.color_palette('deep', 10)
plt.style.use('seaborn-whitegrid')
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)


def week_day_plot_baseline_vs_treatment(baseline: pd.DataFrame,
                                        study: pd.DataFrame,
                                        patient_id: str,
                                        activity_name: str):
    """
    This function is used to plot weekday hourly activity for specific
    subject. This is used to visualize weekly hourly activity pattern for
    baseline and treatment period side by side.
    Parameters
    ----------
    baseline: pd.DataFrame
        Dataframe with second level activity data (index:timestamp,
        timestamp,external_patient_id,label). This is the baseline period second
        level high resolution activity data.
    study: pd.DataFrame
        Dataframe with second level activity data (index:timestamp,timestamp,
        external_patient_id,label). This is the treatment/study period second
        level high resolution activity data.
    patient_id: str
        Patient id is used in naming the title of the plot.
    activity_name: str
        Activity label to visualize (E.g. "Scratching", "Walking", "Running",
        "Resting", "Sleeping", "Shaking")

    Returns
    -------

    """
    fig, ax = plt.subplots(nrows=7, ncols=2, figsize=(8, 12))
    fig.suptitle(patient_id, fontsize=12)
    fig.tight_layout(pad=5)
    for idx, gp in baseline.groupby(baseline.timestamp.dt.dayofweek):
        ax[idx, 0].set_title(gp.timestamp.dt.day_name().iloc[0], color='Black',
                             fontdict={'fontsize': 15})
        data_sample = gp.groupby(gp.timestamp.dt.hour).size().rename_axis(
            'Activity Hourly Baseline').to_frame('') \
            .reindex(np.arange(0, 24, 1)).fillna(0).reset_index()
        data_sample.columns = ['activity_hour', 'activity_baseline']
        scaler = MinMaxScaler()
        data_sample['activity_baseline'] = scaler.fit_transform(
            data_sample['activity_baseline'].values.reshape(-1, 1))
        ax[idx, 0].bar(data_sample['activity_hour'],
                       data_sample['activity_baseline'], color=palette[1])
        ax[idx, 0].set_xlabel(activity_name + ' baseline Hours', fontsize=10)
    for idx, gp in study.groupby(study.timestamp.dt.dayofweek):
        ax[idx, 1].set_title(gp.timestamp.dt.day_name().iloc[0], color='Black',
                             fontdict={'fontsize': 15})
        data_sample = gp.groupby(gp.timestamp.dt.hour).size().rename_axis(
            'Activity Hourly Study').to_frame('') \
            .reindex(np.arange(0, 24, 1)).fillna(0).reset_index()
        data_sample.columns = ['activity_hour', 'activity_treatment']
        scaler = MinMaxScaler()
        data_sample['activity_treatment'] = scaler.fit_transform(
            data_sample['activity_treatment'].values.reshape(-1, 1))
        ax[idx, 1].bar(data_sample['activity_hour'],
                       data_sample['activity_treatment'], color=palette[0])
        ax[idx, 1].set_xlabel(activity_name + ' treatment Hours', fontsize=10)
    plt.show()
