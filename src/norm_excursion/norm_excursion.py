import numpy as np
import pandas as pd
from typing import TypeVar, Tuple
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator

NormExcursion_obj = TypeVar('NormExcursion_obj', bound='NormExcursion')


class NormExcursion(BaseEstimator):
    """ Norm Excursion

    NormExcursion or OneClassSVM is an unsupervised algorithm that learns a
    decision function for novelty detection: classifying new data as similar
    or different to the training set.

    Parameters
    ----------
    patient_id : str
        A parameter used for setting patient id in norm excursion.
        For e.g. "CASE-SAMPLE-01".
    model_hyperparameters: dict
        This parameter is used to give OneClassSVM hyperparameters when training
         norm excursion model on a patient.
         For e.g. {'kernel':'poly',
                  'degree':3,
                  'gamma':'scale',
                  'coef0':0.0,
                  'tol':0.001,
                  'nu':0.5,
                  'shrinking':True,
                  'max_iter':- 1}

    Attributes
    ----------
    patient_id: str
        Get patient ID that is used in norm excursion.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
       """

    def __init__(self, patient_id: str, **model_hyper_parameters: dict):
        self.patient_id = patient_id
        self.model = OneClassSVM(kernel=model_hyper_parameters['kernel'],
                                 degree=model_hyper_parameters['degree'],
                                 gamma=model_hyper_parameters['gamma'],
                                 coef0=model_hyper_parameters['coef0'],
                                 tol=model_hyper_parameters['tol'],
                                 nu=model_hyper_parameters['nu'],
                                 shrinking=model_hyper_parameters[
                                     'shrinking'],
                                 max_iter=model_hyper_parameters['max_iter']
                                 )

    def fit(self, baseline_data: pd.DataFrame) -> NormExcursion_obj:
        """ Implementation of fit function
        One-class-svm is fit on the training data (baseline data or inliners)

        Parameters
        ----------
        baseline_data: pd.DataFrame
            Dataframe with study day variable as index and machine learning
            features as columns.

        E.g
        study_day_variable	Sleep_time	Run_time
                    -20.0	        1.5	      3.0
                    -19.0	        2.0	      3.0
                    -18.0	        2.0	      5.0
                    -17.0	        2.0	      8.0
        Returns
        ----------
        self
            Fitted estimator.

        """
        self.model.fit(baseline_data)
        return self

    def predict(self, treatment_data: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                             pd.DataFrame]:
        """ Implementation of predict function
        We infer treatment period/outlying data points from previously fitted
        model (one-class-svm) in fit() method.

        Parameters
        ----------
        treatment_data: pd.DataFrame
            Dataframe with study day variable as index and machine learning
            features as columns.
        E.g
        study_day_variable	Sleep_time	Run_time
                    0	         1.5	      3.0
                    1	         2.0	      3.0
                    2	         2.0	      5.0
                    3	         2.0	      8.0
        Returns
        ----------
        :rtype: (pd.DataFrame, pd.DataFrame)

        daily_success: pd.DataFrame
            DataFrame with daily outliers/inliner per patient classified as 1/0
            using one-class-svm model in fit() method.
        weekly_success: pd.DataFrame
            DataFrame with weekly proportion of outliers per patient.
        """
        treatment_period_pred = pd.DataFrame(self.model.predict(treatment_data))

        i = 0
        daily_decrease = []
        for g in treatment_period_pred.values:
            i += 1
            # Classifying outliers (-1) as 1 and inliners as 0.
            if g[0] == -1:
                daily_decrease.append([i, 1])
            else:
                daily_decrease.append([i, 0])
        daily_success = pd.DataFrame(daily_decrease)
        daily_success.columns = ['day_name', 'success_days']
        daily_success['external_patient_id'] = self.patient_id

        week_decrease = []
        # Looping through treatment period per week
        for i, g in treatment_period_pred.groupby(
                np.arange(len(treatment_period_pred)) // 7):
            # Qualifying week has 7 days of data
            if g.shape[0] == 7:
                # Proportion of successful (outlying) days per week.
                week_decrease.append([i+1, g[g == -1].count().values[0] / 7])
            else:
                pass
                # Drop the partial week
        weekly_success = pd.DataFrame(week_decrease)
        weekly_success.columns = ['week_name', 'percent_of_success_days']
        weekly_success['external_patient_id'] = self.patient_id
        return daily_success, weekly_success
