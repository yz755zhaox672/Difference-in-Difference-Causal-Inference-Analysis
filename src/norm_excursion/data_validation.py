import pandas as pd


class DataValidation:
    """ Data Validation
    DataValidation class is used to validate the input wearables data extract.
    First part of the validation checks if all the columns are present in
    the extract and second part of the validation checks if the dtypes (of the
    columns) are as expected.

    Parameters
    ----------
    wearables_data: pd.DataFrame
        wearables data extract on which norm excursion analysis will be done.
    """
    def __init__(self, wearables_data: pd.DataFrame):
        self.wearables_data = wearables_data

    def data_validation(self):
        """
        Implementation of data validation function which
        1) Validates if all columns needed are present in the dataframe.
        2) Validates if dtypes of the columns needed match the template
        dataframe.

        Parameters
        ----------
        self

        """
        validation_check_1_bool = False
        print("Validation checks:\n")
        # Column names that need to be present in the dataframe
        columns = ['external_patient_id', 'normalized_study_day',
                   'study_group']
        set_intersection = set(columns).intersection(
            set(self.wearables_data.columns))
        set_difference = set(columns). \
            difference(set(self.wearables_data.columns))
        # Validation 1 : column presence check
        if len(set_intersection) == len(columns):
            print("1) Validation check passes (columns check)\n")
            print("Note: All meta data columns found in extract\n")
            validation_check_1_bool = True
        else:
            print("1) Validation check fails\n")
            print("Note: Below are the missing columns from the extract:\n")
            print(set_difference)
            print("\n")
        # Validation 2 : column dtype check
        if validation_check_1_bool:
            # Create template dataframe with expected dtypes and validate.
            expected_extract_dtypes = pd.DataFrame(
                [['CASE-AA-04', 1.0, 'Purple']],
                columns=['external_patient_id',
                         'normalized_study_day',
                         'study_group']).dtypes
            extract_dtypes = self.wearables_data[columns].dtypes
            if all((expected_extract_dtypes == extract_dtypes).values):
                print("2) Validation check passes (dtypes)\n")
                print("Note: All dtypes match")
            else:
                print("2) Validation check fails (dtypes)\n")
                print("Consider checking dtypes and try again.")
        else:
            print("2) Validation check fails (dtypes)\n")
            print(
                "Note: Consider including missing columns from validation "
                "check (1) and try again.")
