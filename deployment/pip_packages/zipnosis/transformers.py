
####
import logging
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from zipnosis.utils import get_test_note
from zipnosis.setup_logger import logger


from zipnosis.utils import *


logger.debug('runing transformers.py ...')


class ZipnosisNoteTransformer(TransformerMixin):
    '''

    '''
    def __init__(self,
                 input_col=None,
                 output_cols = ('lat', 'lon'),
                 remainder='drop'):

        self.input_col = input_col
        self.output_cols = output_cols
        self.remainder = remainder
        self.feature_names = ('input', 'Section_PatientFreeText',  'Section_TravelFreeText', 'Section_PatientInformation',  'Section_PertinentCovid19Information', 'Section_Symptoms', 'Diagnosis', 'Travel_flag',  'Sex', 'Age',
        'Weight', 'Temperature', 'ICD10', 'ShortnessOfBreath',  'Smokes', 'ContactWithConfirmedCase','ContactWithSuspectedCase'
        'Pregnant', 'ZipCode', 'SymptomQuickness' , 'DaySymptomsStarted', 'cough', 'facial_pain_or_pressure',
        'headache', 'nasal_secretions', 'temperature_symptom', 'sore_throat')

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def _input_handler(self, X):
        input_array = None
        if isinstance(X, list):
            input_array = X
        if isinstance(X, type(np.zeros(0))):
            try:
                input_array = X[: self.input_col]
            except:
                input_array = X.flatten()
        if isinstance(X, type(pd.DataFrame([1]))):
            input_array = X.loc[:, self.input_col].values
        if isinstance(X, type(pd.Series([1]))):
            input_array = X
        if isinstance(input_array, type(None)):
            raise ValueError('unknown input type')
        else:
            return input_array

    def transform(self, X):
        '''
        '''

        input_array = self._input_handler(X)
        df = pd.DataFrame(np.zeros((len(input_array), len(self.feature_names ))),
        columns = self.feature_names)
        anonymized_array =  list(map(anonymize, input_array))

        df.loc[:, 'Section_PatientFreeText'] = list(map(get_patient_free_text, input_array))
        df.loc[:, 'Section_TravelFreeText'] = list(map(get_travel_free_text, input_array))
        df.loc[:, 'Section_PatientInformation'] = list(map(get_patient_information, input_array))
        df.loc[:, 'Section_PertinentCovid19Information'] = list(map( get_pertinent_covid19_information, input_array))
        df.loc[:, 'Section_Symptoms'] = list(map( get_symptoms, input_array))
        df.loc[:, 'Diagnosis'] = list(map(get_diagnosis, input_array))
        df.loc[:, 'Travel_flag'] = list(map(extractTravel, zip(df['Section_PatientFreeText'],df['Section_PertinentCovid19Information'])))
        df.loc[:, 'Sex']  = pd.to_numeric(list(map(extractSex,  df[ 'Section_PatientInformation' ])))
        df.loc[:, 'Age']  =pd.to_numeric( list(map( extractAge,  df[ 'Section_PatientInformation' ])))
        df.loc[:, 'Weight']  = pd.to_numeric(list(map( extractWeight,  input_array)))
        df.loc[:, 'Temperature']  = pd.to_numeric(list(map( extractTemperature, anonymized_array )))
        df.loc[:, 'ICD10']  = list(map( extractICD, anonymized_array ))
        df.loc[:, 'ICDPrefix']  = list(map(extractICDPrefix, anonymized_array  ))
        df.loc[:, 'ShortnessOfBreath']  = pd.to_numeric(list(map(extractShortnessOfBreath, anonymized_array )))
        df.loc[:, 'Smokes']  =pd.to_numeric( list(map(extractSmokes, anonymized_array )))
        df.loc[:, 'ContactWithConfirmedCase']  = pd.to_numeric(list(map(extractLabConfirmedContact, anonymized_array)))
        df.loc[:, 'ContactWithSuspectedCase']  =  pd.to_numeric(list(map( extractSuspectedContact, anonymized_array  )))
        df.loc[:, 'Pregnant']  = pd.to_numeric(list(map(extractPregnancy, anonymized_array  )))
        df.loc[:, 'ZipCode']  = list(map(extractZip, input_array))
        df.loc[:, 'SymptomQuickness']  = pd.to_numeric(list(map(extractSymptomOnsetOrdinal,  anonymized_array )))
        df.loc[:, 'DaySymptomsStarted']  = pd.to_numeric(list(map( extractSymptomStart, anonymized_array )))
        df.loc[:, 'cough']  =  pd.to_numeric(list(map(lambda x: get_symptom(x,'cough'), df['Section_Symptoms'])))
        df.loc[:, 'facial_pain_or_pressure']  =  pd.to_numeric(list(map(lambda x: get_symptom(x,'facial_pain_or_pressure'), df['Section_Symptoms'])))
        df.loc[:, 'headache']  =  pd.to_numeric(list(map(lambda x: get_symptom(x,'headache'), df['Section_Symptoms'])))
        df.loc[:, 'nasal_secretions']  =  pd.to_numeric(list(map(lambda x: get_symptom(x,'nasal_secretions'), df['Section_Symptoms'])))
        df.loc[:, 'temperature_symptom']  =  pd.to_numeric(list(map(lambda x: get_symptom(x,'temperature_symptom'), df['Section_Symptoms'])))
        df.loc[:, 'sore_throat']  =  pd.to_numeric(list(map(lambda x: get_symptom(x,'sore_throat'), df['Section_Symptoms'])))
        if self.remainder == 'passthrough':
            df.index = X.index
            output = pd.concat([X, df], axis=1)
            return output
        try:
            df.index = X.index
            return df
        except:
            return df

    def get_feature_names():
        '''
        retruns self.output_cols (tuple of two strings)
        '''
        return list(self.feature_names)


def _test_zipnosis_transformer():
    df = pd.DataFrame([get_test_note()], columns=['text'], index=['pat1'])
    zt =  ZipnosisNoteTransformer('text', remainder='passthrough')
    zt = zt.fit(df)
    logger.info('ZipnosisNoteTransformer test completed')
