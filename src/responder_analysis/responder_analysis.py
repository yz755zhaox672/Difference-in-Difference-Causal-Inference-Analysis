import numpy as np
import pandas as pd
from typing import TypeVar, Tuple
from sklearn.base import BaseEstimator

NormExcursion_obj = TypeVar('NormExcursion_obj', bound='NormExcursion')


class ResponderAnalysis():
    def __init__(self, feature_col: str, direction: str, success_percent: int):
        self.feature_col=feature_col
        self.direction=direction
        self.success_percent=success_percent
        
    def percentage_change(self, treatment_period: pd.DataFrame, baseline_mean: float):
        # Handling 1/0 condition.
        treatment_period['% '+self.direction+' '+self.feature_col]=((treatment_period[self.feature_col]-baseline_mean)\
                                           /baseline_mean).values*100
        
        return treatment_period
    
    def predict(self, pet_change_matrix: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                             pd.DataFrame]:
        weekly_patient_analysis=[]
        for x in pet_change_matrix.columns:
            flatten_decrease=pd.DataFrame(pet_change_matrix[x].values.flatten())\
                        .dropna().reset_index().drop(['index'],axis=1)
            i=0
            week_decrease=[]
            for i, g in flatten_decrease.groupby(np.arange(len(flatten_decrease)) // 7):
                i+=1
                if g.shape[0]==7:
                    if self.direction=='increase':
                        week_decrease.append([i,g[g>self.success_percent].count().values[0]/7])
                    elif self.direction=='decrease':
                        week_decrease.append([i,g[g<-self.success_percent].count().values[0]/7])
            weekly_success=pd.DataFrame(week_decrease)
            weekly_success.columns=['week_name','percent_of_success_days']
            weekly_success['external_patient_id']=x
            weekly_patient_analysis.append(weekly_success)

        weekly_success_df=pd.concat(weekly_patient_analysis).pivot(index='week_name', columns='external_patient_id', \
                                        values='percent_of_success_days')
        
        return weekly_success_df