class StudyMetaData:
    """ Study Meta Data

    Study meta data holds meta information like study name, sampling_criteria,
    group_1 name and group_2 name.

    Parameters
    ----------
    study_name : str
        Name of the study.
    sampling_criteria : list
        List of patients to be included in the study.
        For e.g. ['CASE-SAMPLE-01','CASE-SAMPLE-02']
    group_1: str
        group 1 name in the study.
    group_2: str
        group 2 name in the study.
    color_1: tuple
        group 1 color code (RGB)
        For e.g. (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    color_2: tuple
        group 2 color code (RGB)
        For e.g. (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    """

    def __init__(self, study_name: str, sampling_criteria: list,
                 group_1: str, group_2: str,
                 color_1: tuple, color_2: tuple):
        self.study_name = study_name
        self.sampling_criteria = sampling_criteria
        self.group_1 = group_1
        self.group_2 = group_2
        self.color_1 = color_1
        self.color_2 = color_2
