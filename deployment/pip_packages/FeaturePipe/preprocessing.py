from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse import coo_matrix, hstack, csc_matrix, csr_matrix, vstack
import numpy as np
import logging
import pandas as pd
import re
from itertools import chain
from FeaturePipe import setup_logger
logger = logging.getLogger('FeaturePipe.preprocessing')


#############
# cleaning functions
###############

class MedicalCleaner(BaseEstimator, TransformerMixin):
    '''
    Class for cleaning string input data to floats
    handles 110/80 type blood preasure columns and returns diastolic an systolic columns
    normalizes temp and weight data based on dectection of units

    usage:
        df = pd.DataFrame({'bp': ['120/90', '150/20', None],
                      'bp_2': ['120/90', '150/20', 1],
                      'temp': ["36.8 C (98.2 F)", "36.8C", '98.2'],
                      'w': ['83.0 lbs 1.2 oz', '10kg', 123],
                      'to_float': ['11 MMOL/L', '>59 mL/min/1.73 sq.m', None],
                      'to_float2': ['11 MMOL/L', '>59 mL/min/1.73 sq.m', 1]
                      })

        col_dict = {'bpCols': ['bp', 'bp_2'], 'tempCols': ['temp'], 'weightCols': ['w'], 'floatCols': ['to_float', 'to_float2']}

        c = MedicalCleaner(**col_dict, fillValue=0)
        results = c.fit_transform(df)
        results = pd.DataFrame(results.toarray(), columns=c.get_feature_names())
        print(results)
    '''


    def __init__(self,  bpCols =None, floatCols=None, tempCols=None, weightCols=None, fillValue=0, sparse=True):
        '''

        :param bpCols: list of blood pressure columns of type "110/80"
        :param floatCols: list of columns to coerce from string to float (remove characters)
        :param tempCols: list of temp cols of type "90 F"
        :param weightCols: list of weight cols " 10lbs 9oz" ( normalized to lbs)
        :param fillValue: numeric value to fill nulls
        :param sparse: maintain sparse array as output
        '''

        self.feature_names = None
        self.bpCols = bpCols
        self.floatCols = floatCols
        self.tempCols = tempCols
        self.weightCols = weightCols
        self.fillValue = fillValue
        self.sparse = sparse
        #self.getInputCols()


    def fit(self, X, y=None):
        '''

        :param X: Panda Data Frame ( contains all input columns)
        :param y: unused array (to maintaine compatability with sklearn pipelines
        :return: self
        '''
        self._validateInput(X)
        return self

    def transform(self, X, y=None):
        '''

        :param X: Panda Data Frame
        :param y: unused
        :return: scipy sparse array of transformed numeric features
        '''
        output_list = []
        if self.bpCols:
            output_list = output_list + list(self._bp_sys_col_gen(X))
            output_list = output_list + list(self._bp_dia_col_gen(X))
        if self.floatCols:
            output_list = output_list + list(self._float_col_gen(X))
        if self.weightCols:
            output_list = output_list + list(self._weight_col_gen(X))
        if self.tempCols:
            output_list = output_list + list(self._temp_col_gen(X))
        output = hstack(output_list)
        if self.sparse:
            return output
        else:
            return output.toarray()

    def _bp_sys_col_gen(self, X):
        for col in self.bpCols:
            logger.debug('running bp cols method for {}'.format(col))
            yield csc_matrix(list(map(lambda v: bp(v, systolic=True, na_val=self.fillValue), X[col]))).transpose()

    def _bp_dia_col_gen(self, X):
        for col in self.bpCols:
            logger.debug('running bp cols method for {}'.format(col))
            yield csc_matrix(list(map(lambda v: bp(v, systolic=False, na_val=self.fillValue), X[col]))).transpose()

    def _weight_col_gen(self, X):
        for col in self.weightCols:
            logger.debug('running bp cols method for {}'.format(col))
            yield csc_matrix(list(map(lambda v: weight(v, na_val=self.fillValue), X[col]))).transpose()

    def _float_col_gen(self, X):
        for col in self.floatCols:
            logger.debug('running bp cols method for {}'.format(col))
            yield csc_matrix(list(map(lambda v: string_to_float(v, na_val=self.fillValue), X[col]))).transpose()

    def _temp_col_gen(self, X):
        for col in self.tempCols:
            logger.debug('running bp cols method for {}'.format(col))
            yield csc_matrix(list(map(lambda v: temp(v, c=False, na_val=self.fillValue), X[col]))).transpose()


    def get_feature_names(self):
        '''
        Helper method to be constitant with sklearn pipelines
        :return:
        '''
        feature_names = []
        if self.bpCols:
            for col in self.bpCols:
                feature_names = feature_names + [col + '_sys']
            for col in self.bpCols:
                feature_names = feature_names + [col + '_dia']
        if self.floatCols:
            feature_names = feature_names + self.floatCols
        if self.weightCols:
            feature_names = feature_names + self.weightCols
        if self.tempCols:
            feature_names = feature_names + self.tempCols
        return feature_names

    def getInputCols(self):
        inputCols = set()
        if self.bpCols:
            inputCols.update(self.bpCols)
        if self.floatCols:
            inputCols.update(self.floatCols)
        if self.weightCols:
            inputCols.update(self.weightCols)
        if self.tempCols:
            inputCols.update(self.tempCols)
        return inputCols

    def _validateInput(self, X):
        missing = self.getInputCols().difference(set(X.columns))
        if missing:
            logger.error('missing input columns: {}'.format(missing))
        else:
            logger.debug('input column names validated')
    # create test data


    # fit and transform method
def bp(x, systolic=True, na_val=0):
    '''
    Blood Pressure Cleaner
    systolic BP, mmHg / 	diastolic BP, mmHg
    to float

    :param x:
    :param systolic:
    :param na_val:
    :return:
    '''
    if x is None:
        output = na_val
    else:
        x = str(x)
        float_pattern = '(?=[^\d])*\d+[.]?\d*'
        x = re.sub('\s{2,}', ' ', x).lower()
        val_list = re.findall(float_pattern, x)
        n_vals = len(val_list)
        if len(val_list) != 2:
            logger.debug('preprocessing.py bp clearner found {} values, should have found 2'.format(n_vals))
        try:
            if systolic:
                output = float(val_list[0])
                logger.debug('preprocessing.py bp_cleaner using systolic ')
            else:
                output = float(val_list[1])
                logger.debug('preprocessing.py bp_cleaner using diastolic ')
            if all((output > 0, output < 700)):
                logger.debug('preprocessing.py bp_cleaner input value validated')
            else:
                logger.debug('preprocessing.py bp_cleaner got invalued values {output}, returning na_val')
                output = na_val
        except IndexError:
            output = na_val
        except TypeError:
            output = x
    return output


def temp(x, c=False,  na_val=0):
    '''
    Function for find valid Patient temperatures in strings

    :param x: string temp, ex: "36.8 C (98.2 F)"
    :param c: logical, return Celsius
    :param na_val: value to return when no valid temperatures are found
    :return: floating tempurature , returns 0 if temp out of ranges
        Celsius,  all((v > 20.0, v < 40.0))
        Fahrenheit ,   all((v > 70.0, v < 120.0))
        returns first tempurature found

        useage: tempurature_cleaner( "36.8 C (98.2 F)", c=False)
    '''
    # replaces two or more white spaces with one
    x = string_to_float(x, na_val=na_val)
    if x:
        if all((x >= 50.0, x < 130.0)):
            if c:
                logger.debug('temp cleaner  converting vals to F -> C')
                output = (x - 32) / 1.8
            else:
                logger.debug('temp cleaner found Farienhiet values ')
                output = x
        elif all((x > 0.0, x < 50.0)):
            if c:
                logger.debug('temp cleaner found ceclius values ')
                output = x
            else:
                logger.debug('temp cleaner converting vals to C -> F')
                output = x * 1.8 + 32
        else:
            logger.debug('temp cleaner temp value {} outside of valid range 0 - 130')
            output = na_val
    else:
        output = na_val
        logger.debug('temp cleaner no tempture values found ')
    return output


def weight(x, kg=False, na_val=0):
    '''

    :param x: string wieghts, ex:  "53.1 kg (117 lb)"
    :param kg: logical, return kg, if False returns lbs
    :return:
        float lbs or kgs
        system is smart enough to handel adding oz to lbs when returning lbs
        usage:
    '''
    try:
        output = float(x)
    except ValueError:
        float_pattern = '(?=[^\d])*\d+[.]?\d*'
        x = re.sub('\s{2,}', ' ', x).lower()
        output = na_val
        # case where kgs are found
        try:
            pattern = float_pattern + '(?=.k|k)'
            v = re.findall(pattern, x)
            logger.debug('weight cleaner found {} kg value'.format(v))
            v = float(v[0])
            if kg is False:
                output = v * 2.204623
                logger.debug('weight cleaner converting g -> lb')
            else:
                output = v
        except IndexError:
            # case where grams are found
            try:
                pattern = float_pattern + '(?=.g|g)'
                v = re.findall(pattern, x)
                logger.debug('weight cleaner found {} g values'.format(v))
                v = float(v[0])
                if kg:
                    output = v/1000
                    logger.debug('weight cleaner converting g -> kg')
                else:
                    output = v * 0.002204623
                    logger.debug('weight cleaner converting g -> lb')
            except:
                # case where lbs are found
                try:
                    pattern = float_pattern + '(?=.l|l)'
                    v = re.findall(pattern, x)
                    logger.debug('weight cleaner found {} lbs values'.format(v))
                    v_lbs = float(v[0])
                    # case where there are lbs and oz specified
                    try:
                        pattern = float_pattern + '(?=.o|o)'
                        v_oz = re.findall(pattern, x)
                        logger.debug('weight cleaner found {} oz values'.format(v_oz))
                        v_oz = float(v_oz[0]) * 0.0625
                        v_lbs = v_lbs + v_oz
                    except IndexError:
                        logger.debug('weight cleaner no oz values found ')
                    if kg:
                        output = v_lbs * 2.204623
                        logger.debug('weight cleaner converting lbs -> kg')
                    else:
                        output = v_lbs

                except IndexError:
                    #case where only oz are specified
                    try:
                        pattern = float_pattern + '(?=.o|o)'
                        v = re.findall(pattern, x)
                        logger.debug('weight cleaner found {} oz values'.format(v))
                        output = float(v[0])
                    except IndexError:
                        logger.debug('no oz values found')
        else:
            if all((output >= 0, output < 551)):
                logger.debug('weight cleaner value validated')
            else:
                logger.debug('weight cleaner unrealistic value {} lbs '.format(output))
                output = na_val
    except TypeError:
        output = na_val
    return output


def string_to_float(x, na_val=0):
    '''
    get a floating number with optional decimal from a string
    :param x: string, float or int
    :param x: string, float int or None, value returned when no digits are found
    :return: string
    '''
    try:
        output = float(x)
    except:
        try:
            pattern = '(?=[^\d])*\d+[.]?\d*'
            v = re.findall(pattern, x)
            output = float(v[0])
        except:
            output = na_val
    return output


def _medicalCleanerTest():

    df = pd.DataFrame({'bp': ['120/90', '150/20', None],
                      'bp_2': ['120/90', '150/20', 1],
                      'temp': ["36.8 C (98.2 F)", "36.8C", '98.2'],
                      'w': ['83.0 lbs 1.2 oz', '10kg', 123],
                      'to_float': ['11 MMOL/L', '>59 mL/min/1.73 sq.m', None],
                      'to_float2': ['11 MMOL/L', '>59 mL/min/1.73 sq.m', 1]
                      })

    col_dict = {'bpCols': ['bp', 'bp_2'],
    'tempCols': ['temp'],
    'weightCols': ['w'], 'floatCols': ['to_float', 'to_float2']}

    c = MedicalCleaner(**col_dict, fillValue=0)
    results = c.fit_transform(df)
    results = pd.DataFrame(results.toarray(), columns=c.get_feature_names())
    print(results)
