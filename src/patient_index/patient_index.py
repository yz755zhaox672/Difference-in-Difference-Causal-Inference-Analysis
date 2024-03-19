# import data packages
import numpy as np
import pandas as pd
from typing import TypeVar
from matplotlib.pylab import plt
# import data preprocessing packages
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
# import metrics from sklearn
from sklearn.metrics import confusion_matrix, classification_report, \
    roc_auc_score, precision_score, recall_score, accuracy_score
# import machine learning algorithms
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

PatientIndex_obj = TypeVar('PatientIndex_obj', bound='PatientIndex')


class PatientIndex(BaseEstimator):
    """ Implementing Patient index using machine learning
        Used to develop patient level index to distinguish between patients.
    """

    def __init__(self, model_name, params):
        """

        Parameters
        ----------
        model_name: str
            model name to be used in developing the index.
            Ex: 'xgboost', 'catboost' or 'logistic'.
        params: dict
            Model parameters as dictionary.
        """
        self.model_name = model_name
        self.y_train_pred = None
        self.y_test_pred = None
        self.y_outsample_pred = None
        self.train_pred_prob = None
        self.test_pred_prob = None
        self.outsample_pred_prob = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_test_outsample = None
        self.y_test_outsample = None

        if model_name == 'xgboost':
            self.model = XGBClassifier(**params)
        elif model_name == 'catboost':
            self.model = CatBoostClassifier(**params)
        elif model_name == 'logistic':
            self.model = LogisticRegression(**params)

    @staticmethod
    def time_series_split(activity_data, date_col_name,
                          pet_col_name, label_col_name,
                          features_list, test_size):
        """ Implements time series data split
        Used to split activity data by time (data column). returns train and
        out of sample test datasets.

        Parameters
        ----------
        activity_data: pd.DataFrame
            Activity data is the daily descriptive statistical features data
            used to develop patient index.
        date_col_name: str
            Name of the column which represents date in activity data.
        pet_col_name: str
            Name of the column which represents pet name in activity data.
        label_col_name: str
            Name of the column which represents target label in activity data.
        features_list: list
            List of the features to be included while dividing the data into
            train and test (out of sample test) sets.
        test_size: float
            Percentage of data to be taken as test set.
            Ex: test_size=30
                This sets 30% of data as out of sample test set.

        Returns
        -------
        :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
        x_train_set: pd.DataFrame
            Input training dataset
        y_train_set: pd.DataFrame
            target labels for training dataset
        x_test_outsample: pd.DataFrame
            Input testing dataset (out of sample set)
        y_test_outsample: pd.DataFrame
            target labels for testing dataset

        """

        activity_data_sort = activity_data.sort_values([date_col_name])

        x = activity_data_sort[features_list + [pet_col_name, date_col_name]]
        y = activity_data_sort[label_col_name].values

        test_set = 100 - test_size
        x_train_set = x.iloc[:int(x.shape[0] / 100 * test_set)]
        x_test_outsample = x.iloc[int(x.shape[0] / 100 * test_set):]

        y_train_set = y[:int(x.shape[0] / 100 * test_set)]
        y_test_outsample = y[int(x.shape[0] / 100 * test_set):]

        return x_train_set, y_train_set, x_test_outsample, y_test_outsample

    @staticmethod
    def split_by_animal(activity_data, features_list,
                        group_1_outsample, group_2_outsample,
                        pet_col_name, date_col_name, label_col_name):
        """ Implements data split by animal (patient)
        This function splits activity data by patient (animal).

        Parameters
        ----------
        activity_data: pd.DataFrame
            Activity data is the daily descriptive statistical features data
            used to develop patient index.
        features_list: list
            List of the features to be included while dividing the data into
            train and test (out of sample test) sets.
        group_1_outsample: list
            List of animals (patients) to be taken as out of sample test set in
            group 1.
        group_2_outsample: list
            List of animals (patients) to be taken as out of sample test set in
            group 2.
        pet_col_name: str
            Name of the column which represents pet name in activity data.
        date_col_name: str
            Name of the column which represents date in activity data.
        label_col_name: str
            Name of the column which represents target label in activity data.

        Returns
        -------
        :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
        x_train_set: pd.DataFrame
            Input training dataset
        y_train_set: pd.DataFrame
            target labels for training dataset
        x_test_outsample: pd.DataFrame
            Input testing dataset (out of sample set)
        y_test_outsample: pd.DataFrame
            target labels for testing dataset

        """
        x_train_set = activity_data[features_list +
                                    [pet_col_name, date_col_name,
                                     label_col_name]]
        x_train_set = x_train_set[~x_train_set[pet_col_name]
            .isin(group_1_outsample + group_2_outsample)]
        y_train_set = x_train_set[label_col_name]

        x_train_set = x_train_set.drop([label_col_name], axis=1)

        x_test_outsample = activity_data[
            features_list + [pet_col_name, date_col_name, label_col_name]]
        x_test_outsample = x_test_outsample[x_test_outsample[pet_col_name].isin(
            group_1_outsample + group_2_outsample)]
        y_test_outsample = x_test_outsample[label_col_name]

        x_test_outsample = x_test_outsample.drop([label_col_name], axis=1)
        return x_train_set, y_train_set, x_test_outsample, y_test_outsample

    @staticmethod
    def class_imbalance(x, y, params):
        """ Implements Random over sampler
        Used to address class imbalance issue using random over sampler.

        Parameters
        ----------
        x: array
            Input data which contains pet activity.
        y: array
            target label of pet activity.
        params: dict
            RandomOverSampler parameters as a dict.

        Returns
        -------
        :rtype: (array, array)
        x_over: array
            Oversampled input data which contains pet activity.
        y_over: array
            Oversampled target label data which contains pet activity.

        """
        oversample = RandomOverSampler(**params)
        x_over, y_over = oversample.fit_resample(x, y)

        return x_over, y_over

    @staticmethod
    def generate_smooth_features(rolling_window, activity_data, pet_col_name,
                                 date_col_name, label_col_name, true_label_dict,
                                 cols_range):
        """ Implements smooth features generation
        Used to generate smooth features using rolling window parameter and
        activity data.

        Parameters
        ----------
        rolling_window: int
            Rolling window period on which the smooth features are generated.
        activity_data: pd.DataFrame
            Activity data is the daily descriptive statistical features data
            used to develop patient index.
        pet_col_name: str
            Name of the column which represents pet name in activity data.
        date_col_name: str
            Name of the column which represents date in activity data.
        label_col_name: str
            Name of the column which represents target label in activity data.
        true_label_dict: dict
            This contains the true label of patient.
            Example: {'Sample_dog':1, 'another_dog':0}
        cols_range: list
            Range of columns to run through smooth feature generation.


        Returns
        -------
        :rtype: pd.DataFrame
        smooth_feature_df: pd.DataFrame
            This dataframe contains smooth features which is returned.
        """
        smooth_features_append = []
        for x in activity_data[pet_col_name].unique():
            one_pet = activity_data[activity_data[pet_col_name] == x] \
                .sort_values([date_col_name])
            smooth_pet = one_pet[one_pet.columns[cols_range[0]:cols_range[1]]] \
                .rolling(rolling_window).mean().dropna()
            smooth_pet[pet_col_name] = x
            smooth_pet[date_col_name] = \
                one_pet.iloc[rolling_window - 1:][date_col_name].values
            smooth_pet[label_col_name] = true_label_dict[x]
            smooth_features_append.append(smooth_pet)
        smooth_feature_df = pd.concat(smooth_features_append)
        return smooth_feature_df

    def fit(self) -> PatientIndex_obj:
        """ Implements machine learning algorithm fit() method
        Used to fit the machine learning algorithm using training dataset and
        updating the prediction parameters using model predictions.

        Returns
        -------
        self

        """
        if self.model_name == 'xgboost':
            self.model.fit(self.x_train, self.y_train)
        elif self.model_name == 'catboost':
            train_pool = Pool(self.x_train, self.y_train)
            validate_pool = Pool(self.x_test, self.y_test)
            self.model.fit(train_pool, eval_set=validate_pool)
        elif self.model_name == 'logistic':
            self.model.fit(self.x_train, self.y_train)

        self.y_train_pred = self.model.predict(self.x_train)
        self.y_test_pred = self.model.predict(self.x_test)
        self.y_outsample_pred = self.model.predict(self.x_test_outsample)

        self.train_pred_prob = self.model.predict_proba(self.x_train)[:, 1]
        self.test_pred_prob = self.model.predict_proba(self.x_test)[:, 1]
        self.outsample_pred_prob = self.model.predict_proba(
            self.x_test_outsample)[:, 1]

        return self

    def set_params(self, x_train, y_train, x_test, y_test, x_test_outsample,
                   y_test_outsample):
        """ Implements set params
        Sets the input training, test and out sample data parameters into
        patient index for reuse.
        Parameters
        ----------
        x_train: array
            Input training dataset.
        y_train: array
            Target label training dataset.
        x_test: array
            Input testing dataset.
        y_test: array
            Target label testing dataset.
        x_test_outsample: array
            Input out of sample test dataset.
        y_test_outsample: array
            Target label out of sample test dataset.

        Returns
        -------
        self
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_test_outsample = x_test_outsample
        self.y_test_outsample = y_test_outsample

        return self

    def get_metrics(self, print_opt):
        """ Implements get metrics function
        Used to print/return metrics generated in patient index on train, test
        and out of sample test datasets.

        Parameters
        ----------
        print_opt: bool
            If true, the function prints machine learning classifier metrics
            for train, test sand out of sample test datasets.

            If false, the function returns precision, recall and accuracy score.

        Returns
        -------
        self
        """
        if print_opt:
            print("Train dataset metrics:\n")
            print(confusion_matrix(self.y_train, self.y_train_pred))
            print(classification_report(self.y_train, self.y_train_pred))
            print(
                roc_auc_score(self.y_train, self.train_pred_prob))
            print("Test dataset metrics:\n")
            print(confusion_matrix(self.y_test, self.y_test_pred))
            print(classification_report(self.y_test, self.y_test_pred))
            print(
                roc_auc_score(self.y_test, self.test_pred_prob))
            print("Out of sample test dataset metrics:\n")
            print(
                confusion_matrix(self.y_test_outsample, self.y_outsample_pred))
            print(classification_report(self.y_test_outsample,
                                        self.y_outsample_pred))
            print("Out of sample test dataset auc:\n")
            print(
                roc_auc_score(self.y_test_outsample, self.outsample_pred_prob))
            print("\n")
        else:
            return [
                precision_score(self.y_test_outsample, self.y_outsample_pred),
                recall_score(self.y_test_outsample, self.y_outsample_pred),
                accuracy_score(self.y_test_outsample, self.y_outsample_pred)]

    def index(self, x_train, x_test, x_test_outsample, pet_col_name,
              date_col_name):
        """ Implements patient index creation
            Used to develop patient level index.
        Parameters
        ----------
        x_train: array
            Input training dataset which contains activity data.
        x_test: array
            Input testing dataset which contains activity data.
        x_test_outsample: array
            Input outsample testing dataset which contains activity data.
        pet_col_name: str
            Name of the column which represents pet name in activity data.
        date_col_name: str
            Name of the column which represents date in activity data.

        Returns
        -------
        :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        data_matrix_train: pd.DataFrame
            Train data matrix
        data_matrix_test: pd.DataFrame
            Test data matrix
        data_matrix_outsample: pd.DataFrame
            Test outsample matrix
        """
        # Train index
        train_probabilities = self.model.predict_proba(x_train[:, :-2])[:, 1]
        train_predictions_df = pd.DataFrame(
            {"train_probabilities": train_probabilities,
             date_col_name: x_train[:, -1],
             pet_col_name: x_train[:, -2]})

        data_matrix_train = pd.DataFrame(
            index=pd.DataFrame(train_predictions_df[date_col_name].unique())[0]
                .sort_values().values,
            columns=train_predictions_df[pet_col_name].unique())
        for index, x in train_predictions_df.iterrows():
            data_matrix_train.loc[x[date_col_name]][x[pet_col_name]] = x[
                'train_probabilities']

        # Test index
        test_probabilities = self.model.predict_proba(x_test[:, :-2])[:, 1]
        test_prediction_df = pd.DataFrame(
            {"test_probabilities": test_probabilities,
             date_col_name: x_test[:, -1],
             pet_col_name: x_test[:, -2]})
        data_matrix_test = pd.DataFrame(
            index=pd.DataFrame(test_prediction_df[date_col_name]
                               .unique())[0].sort_values().values,
            columns=test_prediction_df[pet_col_name].unique())
        for index, x in test_prediction_df.iterrows():
            data_matrix_test.loc[x[date_col_name]][x[pet_col_name]] = x[
                'test_probabilities']

        # Out of sample index
        test_outsample_probabilities = self.model.predict_proba(
            x_test_outsample.values[:, :-2])[:, 1]
        test_outsample_prediction_df = pd.DataFrame(
            {"test_outsample_probabilities": test_outsample_probabilities,
             date_col_name: x_test_outsample.values[:, -1],
             pet_col_name: x_test_outsample.values[:, -2]})

        data_matrix_outsample = pd.DataFrame(
            index=pd.DataFrame(test_outsample_prediction_df[date_col_name]
                               .unique())[0].sort_values().values,
            columns=test_outsample_prediction_df[pet_col_name].unique())
        for index, x in test_outsample_prediction_df.iterrows():
            data_matrix_outsample \
                .loc[x[date_col_name]][x[pet_col_name]] = x[ \
                'test_outsample_probabilities']

        return data_matrix_train, data_matrix_test, \
               data_matrix_outsample

    def index_split_by_animal(self, x_train, x_test, x_test_outsample,
                              pet_col_name, date_col_name):
        """ Implements patient index creation
            Used to develop patient level index for split by animal
            methodology.
        Parameters
        ----------
        x_train: array
            Input training dataset which contains activity data.
        x_test: array
            Input testing dataset which contains activity data.
        x_test_outsample: array
            Input outsample testing dataset which contains activity data.
        pet_col_name: str
            Name of the column which represents pet name in activity data.
        date_col_name: str
            Name of the column which represents date in activity data.

        Returns
        -------
        :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        data_matrix_train_test: pd.DataFrame
            Train and test data matrix
        data_matrix_outsample: pd.DataFrame
            Test outsample matrix
        """
        # Train index
        train_probabilities = self.model.predict_proba(x_train[:, :-2])[:, 1]
        train_predictions_df = pd.DataFrame({"probabilities": \
                                                 train_probabilities,
                                             date_col_name: x_train[:, -1],
                                             pet_col_name: x_train[:, -2]})
        # Test index
        test_probabilities = self.model.predict_proba(x_test[:, :-2])[:, 1]
        test_prediction_df = pd.DataFrame({"probabilities": test_probabilities,
                                           date_col_name: x_test[:, -1],
                                           pet_col_name: x_test[:, -2]})

        train_test_df = pd.concat([train_predictions_df, test_prediction_df])
        data_matrix_train_test = pd.DataFrame(
            index=pd.DataFrame(train_test_df[date_col_name].unique())[
                0].sort_values().values,
            columns=train_test_df[pet_col_name].unique())
        for index, x in train_test_df.iterrows():
            data_matrix_train_test.loc[x[date_col_name]][x[pet_col_name]] = \
                x['probabilities']

        # Out of sample index
        test_outsample_probabilities = self.model.predict_proba(
            x_test_outsample[:, :-2])[:, 1]
        test_outsample_prediction_df = pd.DataFrame(
            {"test_outsample_probabilities": test_outsample_probabilities,
             date_col_name: x_test_outsample[:, -1],
             pet_col_name: x_test_outsample[:, -2]})

        data_matrix_outsample = pd.DataFrame(
            index=pd.DataFrame(test_outsample_prediction_df[date_col_name]
                               .unique())[0].sort_values().values,
            columns=test_outsample_prediction_df[pet_col_name].unique())
        for index, x in test_outsample_prediction_df.iterrows():
            data_matrix_outsample.loc[x[date_col_name]][x[pet_col_name]] = x[
                'test_outsample_probabilities']

        return data_matrix_train_test, data_matrix_outsample

    @staticmethod
    def remove_correlated_features(data, threshold):
        """ Implements remove correlated features
        Used to remove highly correlated features.

        Parameters
        ----------
        data: pd.DataFrame
            Activity dataframe with daily patient activity.
        threshold: float
            Threshold to remove outliers.

        Returns
        -------
        correlated_removed_features: list
            correlated features removed list.

        """
        correlation_matrix = data.corr()
        # Select upper triangle of correlation matrix
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

        pair_correlation = pd.DataFrame(upper.stack()).reset_index()
        pair_correlation.columns = ['col1', 'col2', 'correlation']

        drop_corr_vars = set(list(
            pair_correlation[pair_correlation['correlation'] > threshold][
                'col1'].values))

        correlated_removed_features = list(
            set(data.columns).difference(drop_corr_vars))

        return correlated_removed_features

    @staticmethod
    def select_optimal_features(x_train_set, y_train_set, columns):
        scaler = MinMaxScaler()
        # Create the RFE object and compute a cross-validated score.
        lr = LogisticRegression(class_weight='balanced', random_state=21)
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=lr, step=1, cv=StratifiedKFold(5),
                      scoring='roc_auc')
        rfecv.fit(scaler.fit_transform(x_train_set), y_train_set)
        features_selected_recursive = pd.DataFrame(
            pd.Series(dict(zip(columns[:-2], rfecv.support_))))
        print("Optimal number of features : %d" % rfecv.n_features_)
        print(features_selected_recursive)
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        return features_selected_recursive
