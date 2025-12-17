from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import pandas as pd
import numpy as np
import logging
from warnings import warn
from FeaturePipe import setup_logger
from FeaturePipe.feature_extraction import Text2List
logger = logging.getLogger('FeaturePipe.pipeline')


def col_sub_setter(data, cols=None, feature_names=None, return_array=False):
    '''
    :param data: pandas data frame or numpy array
    :param cols: list or str int of columns
    :param feature_names list of feature names, same length as data.shape[1]
    :return: pandas or numpy array

    '''
    # the case when input data is pandas

    if type(cols) is str:
        cols = [cols]
    elif type(cols) is int:
        cols = [cols]
    else:
        pass
    if hasattr(data, 'columns'):
        logger.debug('using pandas sub setting method')
        missing = set(cols).difference(data.columns)
        if missing:
            logger.warning('sub setting missing {0} columns from input data'.format(cols))
        else:
            logger.debug('sub setting column check passed')
        try:
            output = data.iloc[:, cols]
        except TypeError:
            output = data.loc[:, cols]
        output.columns = cols
    # the case when input data is numpy
    else:
        # where selecting names from a list of names
        if feature_names is not None:
            logger.debug('using numpy sub setting method')
            logger.debug('col sub setter using feature names')
            feature_dict = dict(list(enumerate(feature_names)))
            feature_dict = {v: k for k, v in feature_dict.items()}
            if len(feature_names) == data.shape[1]:
                logger.debug('feature names matches data shape[1] {0}'.format(len(feature_names)))
            else:
                logger.error('feature names {0} miss matches data shape[1] {1}'
                             .format(len(feature_names), data.shape[1]))
            cols = [feature_dict[name] for name in cols]
            if return_array:
                output = data[:, cols]
            else:
                output = pd.DataFrame(data[:, cols], columns=feature_names)
        # case where selection is a list of ints
        else:
            try:
                output = data[:, cols]
            except IndexError:
                raise IndexError(' if feature names is none, cols must be int or list of ints')
        if data.shape[0] == output.shape[0]:
            logger.debug('col select output data shape[0] validated')
        else:
            logger.error('data shape {0} output {1}'.format(data.shape[0], output.shape[0]))
        if len(cols) == output.shape[1]:
            logger.debug('n_col and  output data shape[1] validated')
        else:
            logger.error('n_cols: {0}, output n_cols: {1}'.format(len(cols), output.shape[0]))
    return output


class TmmAdaptor(BaseEstimator, TransformerMixin):
    '''
    An adapter to use Data Set Builder from TextMiningMachine in an sklean pipe line.
    '''
    def __init__(self, trans, sparse=False):
        '''
        A class to be called inside sklearn pipeline,
        can be prefitted, or fitted during the pipe.fit
        to use pre fitted, pas in trans= DataSetBuilder() object
        :param col_dict:
            col_dict = {'imputer_cols': []
                'zero_imputer_cols': []
                'standard_scaler_cols': []
                'robust_scaler_cols': []
                'min_max_cols': []
                'cat_cols': []
                'cat_from_text_cols': []
                'single_cat_cols': []
                'text_cols'

        '''
        from itertools import chain
        self.dsb = trans
        self.sparse = sparse
        self.col_dict = self.dsb.col_dict
        self.feature_names = self.dsb.feature_names
        self.input_names = list(chain.from_iterable(self.col_dict.values()))

    def fit(self, X, y=None):
        '''
        Either fits a dataset builder object, or returns a prefitted object if the
        class has init with a trans parameter
        :param X: pandas data frame
        :param y: None
        :return: self
        '''
        return self

    def transform(self, X, y=None):
        '''

        :param X: pandas data frame with columns in input_cols
        :return: scipy sparse matrix of extracted features
        '''
        output = self.dsb.transform(X)
        if self.sparse:
            return output
        else:
            return output.toarray()


class FeatureName:
    '''
    A class to extract feature names.txt out of an sklearn pipeline.
    '''
    def __init__(self):
        pass

    def _feature_names(self, pipe):
        '''
        Internal method to scrape  feature names.txt for an sklearn pipelines
        :param pipe: sklearn pipeline
        :return: generator
        '''
        import re
        logger.info('FeatureName pipe_feature_names called')
        input_names = None
        feature_names = None
        # the case where the input is a feature union
        if hasattr(input, 'steps'):
            logger.info('pipeline input detected to get_feature_names')
            pipe_list = [input]
        else:
            logger.warning(
                'unknown input to get_feature_names, takes sklearn pipeline')

        logger.debug('get pipe names searching for attribute_names in pipe.steps')
        # loops through steps in each pipeline to find input names
        for i, step in enumerate(pipe.steps):
            step = step[1]
            if hasattr(step, 'attribute_names'):  # tries to find input names.txt in Selector Class
                input_names = step.attribute_names
                logger.info('found attribute names : ' + str(input_names))
                if type(input_names) is str:
                    input_names = [input_names]
        logger.debug('get pipe names searching for feature names in reversed pipe steps')
        # loops through each step in the reversed pipeline to find feature names
        for i, step in enumerate(reversed(pipe.steps)):  # starts at the end of the transform works foreward
            step = step[1]
            if feature_names is None:
                if hasattr(step, 'get_feature_names'):  # tries to find sklearn get_feature_names function
                    feature_names = list(step.get_feature_names())
                    logger.info('get pipe names found get_feature_names method in step: ' + str(i))
            if feature_names is None:
                if hasattr(step, 'feature_names'):  # tries to find sklearn get_feature_names function
                    feature_names = step.feature_names
                    logger.info('get pipe names found feature_names attr method in step: ' + str(i))
            if feature_names is None:
                if hasattr(step, 'n_components'):  # the case with numeric pca (n feature names)
                    n_comps = step.n_components
                    feature_names = ['pca' + str(n) for n in range(n_comps)]
                    logger.info('get pipe names found fn_components_ attr, using pca as prefix for feature names,'
                                'step: ' + str(i))
            else:
                logger.debug('no feature names or components found in step: ' +  str(i))
        logger.debug('found input_names: ' + str(input_names))
        logger.debug('found feature names: ' + str(feature_names))

        # case when no input names are found
        if input_names is None:
            logger.warning('input names could not be determined, first step needs an attribute_names attr')
            if feature_names is not None:
                logger.info('no modifications to feature names made')
            else:
                logger.warning('no feature names or input names found in pipeline')
        # case where input names are found
        else:
            if feature_names is None:
                logger.warning('assuming input_names = feature_names')
                feature_names = input_names
            else:
                if set(input_names).difference(set(feature_names)):
                    logger.info('prefixing input names to feature names')
                    for i, f_name in enumerate(feature_names):
                        pattern = '^x\d+'
                        regex_num = re.findall(pattern, f_name)
                        pattern = '^[^[.|_]*'
                        regex_col = re.findall(pattern, f_name)
                        # case when feautre names are of the x0_cat, x1_cat format
                        if regex_num:
                            n = int(re.findall('\d+', regex_num[0])[0])
                            feature_names[i] = re.sub(pattern, input_names[n], f_name)
                        # case when there is only one input col
                        elif len(input_names) == 1:
                                feature_names[i] = input_names[0] + '.' + f_name
                        else:
                            if regex_col:
                                if regex_col[0] in set(input_names):
                                    logging.debug('col prefix {0} found in input cols'.format(regex_col[0]))
                                else:
                                    logger.warning('col prefix {0} NOT found in input cols'.format(regex_col[0]))
                            else:
                                logger.warning('unclear how to associate input names with feature names')
                        logger.debug('f_name before: ' + f_name + ', after: ' + feature_names[i])
                else:
                    logger.info('no difference between input names and feature nmes')
        for f_name in feature_names:
            yield f_name


    def get_feature_names(self, input):
        from itertools import chain
        from warnings import warn
        if hasattr(input, 'transformer_list'):
            pipe_list = [step[1] for step in input.transformer_list]
            logger.info('feature union input to get_feature_names detected')
        elif hasattr(input, 'steps'):
            pipe_list = [input]
            logger.info('pipeline input detected to get_feature_names')
        else:
            warn('input should be an sklearn pipeline or feature union')
            pipe_list = input
        output = list(map(self._feature_names, pipe_list))
        output = [list(gen) for gen in output]
        output = list(chain.from_iterable(output))
        return output

    def _inputs_names(self, pipe):
        '''

        :param pipe:
        :return:
        '''
        output = set()
        if hasattr(pipe, 'steps'):
            for step in pipe.steps:
                if type(step) is tuple:
                    step = step[1]
                if hasattr(step, 'attribute_names'):
                    output = step.attribute_names
                elif hasattr(step, 'input_cols'):
                    output = step.input_cols
                elif hasattr(step, 'col_dict'):
                    output = list(step.col_dict.values())
                else:
                    logger.error('no attribute_names attribute found in pipeline steps, returning empty set')
                if type(output) is str:
                    output = [output]
                output = set(output)
                return output
        else:
            raise ValueError('cannot find attr steps in pipe, or in pipe[1]')

    def get_input_names(self, input):
        from itertools import chain
        from warnings import warn
        if hasattr(input, 'transformer_list'):
            pipe_list = [step[1] for step in input.transformer_list]
            logger.info('feature union input to get_feature_names detected')
        elif hasattr(input, 'steps'):
            pipe_list = [input]
            logger.info('pipeline input detected to get_feature_names')
        else:
            warn('input should be an sklearn pipeline or feature union')
            pipe_list = input
        output = list(map(self._inputs_names, pipe_list))
        output = [list(gen) for gen in output]
        output = set(list(chain.from_iterable(output)))
        return output


class ColumnSelect(TransformerMixin):
    '''
    Select Feature Names from a data frame, to be used inside sklearn pipe lines
    '''

    def __init__(self, col_list=None, dtype=None, fill=None):
        if type(col_list) is str:
            col_list = [col_list]
        if col_list is not None:
            self.attribute_names = col_list
        self.col_list = col_list
        self.dtype = dtype
        self.fill = fill
        self.input_dtypes = {}

    def _set_dtypes(self, X):
        try:
            for _, c in enumerate(self.col_list):
                d = {c: X.dtypes[c]}
                logger.debug(str(d))
                self.input_dtypes.update(d)
            logger.debug('Colselect input dtypes set')
        except AttributeError:
            logger.error('colselect failed to set dtypes')

    def _validate_cols(self, data):
        '''
        ensures that input columns from data frame included the needed columns
        :param data:
        :return:
        '''
        data = pd.DataFrame(data)
        try:
            cols = list(data.columns)
            missing = set(self.attribute_names).difference(set(cols))
            if missing:
                logger.error('ColSelect Class missing expected columns: ', missing)
            else:
                logger.debug('ColSelect class validated input columns')
        except AttributeError:
            logger.warning('Col select class did not find any input column names ')

    def fit(self, X):
        '''

        :param X: data frame
        :return: self
        '''
        try:
            if self.col_list is None:
                self.col_list = list(X.columns)
                self.attribute_names = set(X.columns)
            self._validate_cols(X)
            self._set_dtypes(X)
        except AttributeError:
            logger.debug('input cols not set ')
        return self

    def transform(self, X):
        '''
        A method to select columns from data frames, to be used inside sklearn pipelines
        :param X: a data frame
        :return: data frame with only the columns selected
        '''
        try:
            n_obs = X.shape[0]
        except AttributeError:
            n_obs = None
            logger.error('col select input data has no shape')
        if self.col_list:
            try:
                if X.ndim == 1:
                    X = pd.DataFrame(X).transpose()
                    logger.debug('ColSelect received 1 d input, transposing using pandas')
                    n_obs = X.shape[0]
                else:
                    logger.debug('ColSelect received >1d input')
            except AttributeError:
                logger.debug('ColSelect input data does not have attr ndim ')
            self._validate_cols(X)
            if self.dtype is not None:
                logger.debug('ColSelect forcing dtype to ' + str(self.dtype))
                output = pd.DataFrame(X[self.col_list], columns=self.col_list, dtype=self.dtype)
            else:
                if self.input_dtypes:
                    logger.debug('ColSelect forcing dtype list of dtypes ' + str(self.input_dtypes))
                    output = pd.DataFrame(X[self.col_list], columns=self.col_list)
                    for _, (k, v) in enumerate(self.input_dtypes.items()):
                        logger.debug('setting {0} to {1}'.format(k, v))
                        output[k] = output[k].astype(v)

                else:
                    logger.debug('ColSelect not using dtypes')
                    output = pd.DataFrame(X[self.col_list], columns=self.col_list)
            if self.fill:
                logger.debug('')
                output.fillna(self.fill, inplace=True)
            if output.shape[1] == len(self.col_list):
                logger.debug('ColSelect transform method col check ckeck passed')
            else:
                logger.error('ColSelect transform, have: {0}, needed: {1}'.format(output.shape[0], len(self.col_list)))
        else:
            logger.debug('ColSelecttransform method, not selecting columns, passing through input')
            output = X

        if n_obs:
            if n_obs == output.shape[0]:
                logger.debug('ColSelect transform row check passed')
            else:
                logger.error('ColSelect row check before: {0}, after: {1}'.format(n_obs, output.shape[0]))
        return output

    def get_feature_names(self):
        return self.col_list


class Bagpipe:
    '''
    Desc:  A class to manage xgboost models, pipelines, deciles calculations
    methods to extend to generic models to be included in feature builds
    '''

    def __init__(
            self, model, pipe, data=None, features=None, feature_names=None, input_names=None, name=None, preds=None):
        '''
        A class to manage xgboost models, pipelines, deciles calculations
        :param model: fitted xgboost booster object (model)
        :param pipe: sklearn pipeline object or Feature Union
        :param data: pandas data frame of un transformed data
        :param features: coo matrix of transformed data
        :param feature_names: list of feature names.txt
        :param name: str model name (defaults to init date time)
        :param verbose: logical print debugging
        :param kwargs:
        '''
        from datetime import datetime
        logger = logging.getLogger('FeaturePipe.Bagpipe')

        self.model = model
        self.pipe = pipe
        self.deciles = None
        self.features = features
        self.data = data
        self.feature_names = feature_names
        self.input_names = input_names
        self.steps = None
        self.data = data
        self.features = features
        self.outputs = {}
        self.performance = {'mean': 0, 'std': 0, 'median': 0, 'max': 0, 'min': 0, 'len': 0}
        self.model_init_date = datetime.now()
        # setup a name for the model
        if name is not None:
            self.name = name
        else:
            self.name = 'model :' + str(datetime.now())
            logger.debug('creating mode name: ' + self.name)
        # extract input names from data, if none provided
        if self.input_names is None:
            if data is not None:
                try:
                    self.input_names = set(data.columns)
                    logger.debug('setting input_names from input data ')
                except AttributeError:
                    self.input_names = set(list(range(data.shape[1])))
                    logger.debug('colnames are set using a list of int indexes based on input data.shape[1]')
        else:
            self.input_names = set(self.input_names)
        # extract feature names from the model
        if feature_names is None:
            if hasattr(self.model, 'feature_names'):
                self.feature_names = self.model.feature_names
                logger.debug('setting self.feature_names from the model')
        if preds is not None:
            self._fit_deciles(preds=preds)
            self._set_performance(preds=preds)
        else:
            warn('decilizer is not setup, init with preds arg')
        logger.debug(' bagpipe init completed')

    def _set_performance(self, preds):
        '''
        internal method to keep track of model performance stats
        uses self.outputs['preds'] to set self.performance.update(d)
        :return:
        '''
        try:
            x = preds
            d = {'mean': np.mean(x),
                 'std': np.std(x),
                 'median': np.median(x),
                 'max': np.max(x),
                 'min': np.min(x),
                 'len': len(x)}
            logger.debug('updating self.performance')
            self.performance.update(d)
        except KeyError:
            logger.debug('skippping performance update, self.preds is None')

    def get_pipe_key(self):
        '''
        internal method get that extracts a key (finger print) from an sklearn pipeline
        to be used later to determine when to transform features if multiple models are uses
        :return:
        '''
        output = (set(self.input_names), set(self.feature_names), str(self.pipe))
        if self.input_names is None:
            logger.error(self.name + ', input names.txt not set')
        if self.feature_names is None:
            logger.error(self.name + ', feature names.txt not set')
        if self.pipe is None:
            logger.error(self.name + ', pipe not fitting')
        return output

    def _data_gen(self, data=None, batch_size=100):
        '''
        row
        :param data: pd data frame
        :param batch_size: int
        :return: yields a batch of rows of data, or partial batch in batch_size > remaining rows
        '''
        from FeaturePipe.utils import batch_gen
        self._validate_input(data)
        n_obs = self.data.shape[0]
        gen = batch_gen(n_obs, batch_size)
        logger.debug('data gen init')
        try:
            while True:
                yield self.data.iloc[next(gen), :]
        except StopIteration:
            logger.debug('data gen completed')

    def _transform_gen(self, data=None, batch_size=100):
        '''

        :param data: pd data frame
        :param batch_size: int
        :return:  yields a batch of rows of transformed features, or partial batch in batch_size > remaining rows
        '''
        from scipy.sparse import coo_matrix
        logger.debug('transform gen init')
        self._validate_input(data)
        gen = self._data_gen(batch_size=batch_size)
        try:
            while True:
                temp_data = next(gen)
                output = coo_matrix(self.pipe.transform(temp_data))
                yield output
        except StopIteration:
            logger.debug('transform gen completed')

    def _feature_gen(self, batch_size=100):
        '''
        yields transformed features (self.features are set yields row wise) else runs the self._transform_gen
        to transform self.data and yield features
        :param batch_size: int
        :return:  yields a batch of rows of transformed features, or partial batch in batch_size > remaining rows
        '''
        from scipy.sparse import csr_matrix, coo_matrix
        from FeaturePipe.utils import batch_gen
        logger.debug('init feature gen')
        # uses a generator to yield batches of features, when features have already be extracted
        if self.features is not None:
            logger.debug('using cached features with shape: ' + str(self.features.shape))
            features = csr_matrix(self.features)
            logger.debug('converting self.features to csr')
            n_obs = self.features.shape[0]
            gen = batch_gen(n_obs, batch_size)
            try:
                while True:
                    output = coo_matrix(features[next(gen)])
                    yield output
            except StopIteration:
                logger.debug('_feature gen completed')
        # uses a genertor to transform data and yield features
        else:
            logger.debug(' transform feature gen init')
            gen = self._transform_gen(batch_size=batch_size)
            try:
                while True:
                    yield next(gen)
            except StopIteration:
                logger.debug('transform gen completed ')

    def transform(self, data=None, return_output=True):
        '''
        use the sklearn pipe to
        transform and entire data set, sets the self.features attr
        :param data: pandas data frame
        :param return_output: logical
        :return: scipy sparse matrix
        '''
        from scipy.sparse import vstack
        self._validate_input(data)
        if data is not None:
            logger.debug('transform method running ...')
            logger.debug('data input shape to generator: ' + str(data.shape))
            feature_list = list(self._transform_gen())
            logger.debug('len feature list' + str(len(feature_list)))
            features = vstack(feature_list)
            logger.debug('transform vstacking completed')
            self.features = features
            logger.debug('self.features set from transform method')
            if return_output:
                return features

    def _predict_gen(self, data=None, features=None, pred_contribs=False, batch_size=100):
        '''

        Prediction gen uses provided features, or data, if not are provided use self.features or lastly
        transforms self.data for features
        :param data: pandas data frame
        :param features: scipy sparse matrix
        :param pred_contribs: logical
        :param batch_size: int
        :return: yeilds array of predictions or prediction contributions
        '''
        from scipy.sparse import coo_matrix
        from xgboost import DMatrix
        self._validate_input(data, features=features)
        logger.debug('_predict gen init')
        gen = self._feature_gen(batch_size=batch_size)
        try:
            while True:
                temp_data = next(gen)
                temp_dmatrix = DMatrix(temp_data, feature_names=self.feature_names)
                preds = self.model.predict(temp_dmatrix, pred_contribs=pred_contribs)
                if pred_contribs:
                    preds = coo_matrix(preds)
                yield preds
        except StopIteration:
            logger.debug('_predict gen completed')

    def predict(self, data=None, features=None, pred_contribs=False, batch_size=100, return_output=True):
        '''
        return array of predictions, or sets self.outputs['preds']
        :param data: pandas data frame or numpy array
        :param features: scipy sparse matrix of features
        :param pred_contribs: logical (whether to return shapely contributions
        :param batch_size: int number of rows in a batch
        :param return_output: logical, return output
        :return: numpy array of predictions or prediction contributions
        '''
        from scipy.sparse import vstack
        from itertools import chain
        self._validate_input(data, features=features)
        logger.debug('_predict gen init')
        gen = self._predict_gen(pred_contribs=pred_contribs, batch_size=batch_size)
        if pred_contribs:
            pred_contribs_list = list(gen)
            output = vstack(pred_contribs_list)
        else:
            pred_list = list(gen)
            output = np.array(list(chain.from_iterable(pred_list)))
        if return_output:
            return output

    def _fit_deciles(self, preds, n=100, **kwargs):
        '''
        fits prediction output to a sklearn KBins discretizier based on a uniform distrobution
        :param n: int number of bins (typically 10,or 100)
        :param kwargs: args to the sklearn  KBinsDiscretizer
        :return: None, sets self.deciles
        '''
        import numpy as np
        from sklearn.preprocessing import KBinsDiscretizer
        params = {'encode': 'ordinal', 'strategy': 'quantile'}
        params.update(kwargs)
        preds = np.reshape(np.squeeze(preds), (-1, 1))
        logger.debug('fitting deciles ... with cached preds of len: ' + str(len(preds)))
        d = KBinsDiscretizer(n_bins=n, **params)
        d.fit(preds)
        self.deciles = d
        logger.debug('decile fit completed')

    def predict_deciles(self, preds=None):
        '''

        :param data: pd data frame
        :param features: scipy array of features
        :param preds: array of preds
        :return: array of deciles
        '''
        preds = np.array(preds)
        preds = np.reshape(np.squeeze(preds), (-1, 1))
        logger.debug('transforming deciles with user input preds')
        deciles = self.deciles.transform(preds)
        output = np.array(deciles, dtype=np.int) + 1
        return output

    def _predict_context_gen(self, n=5, batch_size=100, **kwargs):
        '''
        internal method to use contrib feature reducer and summary gen to create a context summary from an xgboost model
        :param n: int number of top features to include in summary
        :param batch_size: int number of rows per batch
        :param kwargs: **kwargs to self._summary_gen
        :return:
        '''
        logger.debug('standard output gen init')
        gen = self._contrib_reducer_gen(n=n, batch_size=batch_size)
        try:
            while True:
                temp_feature_names, temp_features_subset, contributions_subset, temp_preds = next(gen)
                context = list(
                    self._summary_gen(temp_feature_names, temp_features_subset, contributions_subset, temp_preds,
                                      **kwargs))
                yield context
        except StopIteration:
            logger.debug('prediction context gen completed')

    def predict_context(self, data=None, features=None, n=5, batch_size=100, **kwargs):
        '''

        :param data: pandas data frame or numpy array
        :param features: scipy sparse matrix of feature:
        :param n: int number of features to return in text summary
        :param batch_size: int number of rows per batch
        :param kwargs: key word args for _summary_gen
        :return:
        '''
        self._validate_input(data=data, features=features)
        gen = self._predict_context_gen(n=n, batch_size=batch_size, **kwargs)
        conext = list(gen)
        return conext

    def _validate_input(self, data=None, features=None, cols=None):
        '''
        primary internal method for handling inputs,
        :param data:
        :param features:
        :return:
        '''
        logger.debug('validate input method called')
        if data is None:
            if all((self.features is None, self.data is None)):
                raise AttributeError('no data provided, specify data=data in transform method')
            else:
                if features is None:
                    logger.debug('using cached data')
                else:
                    logger.debug('using user input features')
        else:
            if cols is not None:
                if type(cols) is str:
                    cols = [cols]
                    if hasattr(data, 'columns'):
                        missing = set(cols).difference(set(data.columns))
                        if missing:
                            logger.error('data is missing input cols {0}'.format(missing))
                        else:
                            logger.debug('cols in data.columns  check passed')
            self.data = pd.DataFrame(data)
            logger.debug('self.data attr set')
        if self.input_names is None:
            self.input_names = set(self.data.columns)
            logger.debug('setting input_names from input data ')
        else:
            logger.debug(' checking against previous input names.txt')
        missing = self.input_names.difference(set(self.data.columns))
        if missing:
            logger.error('input names.txt could not be validated, missing : ' + str(missing))
            raise ValueError('data is missing input cols: ' + str(missing))
        else:
            logger.debug('input names.txt validated')
        if all((features is None, data is not None)):
            # deletes previous features
            self.features = None
            logger.debug('previous features removed')
        if features is not None:
            self.features = features
            logger.debug('new feature input from user')
        if all((self.features is not None, self.data is not None)):
            if self.features.shape[0] == self.data.shape[0]:
                logger.debug('features and data row check passed')
            else:
                logger.error('features' + str(self.features.shape[0]) + 'and data rows' + str(self.data.shape[0]) +
                              ' miss matched')
        else:
            logger.debug('skipping row check method due to features or data being None')
            logger.debug('validate input completed')

    def _contrib_reducer_gen(self, n=5, batch_size=100):
        from xgboost import DMatrix
        logger.debug('_contrib_reducer_gen  init')
        gen = self._feature_gen(batch_size=batch_size)
        try:
            while True:
                temp_features = next(gen)
                # make d matrix data set
                dmatrix = DMatrix(temp_features, feature_names=self.feature_names)
                # predict data
                temp_preds = self.model.predict(dmatrix, pred_contribs=False)
                # predict feature contributions
                temp_contributions = self.model.predict(dmatrix, pred_contribs=True)
                # make sparse feature dense
                temp_features = temp_features.toarray()
                # sort
                index = np.argpartition(np.abs(temp_contributions[:, :-1]), -n)[:, -n:]
                contributions_subset = np.squeeze([[temp_contributions[i, :][index[i, :]]]
                                                   for i in range(temp_contributions.shape[0])])
                temp_features_subset = np.squeeze([[temp_features[i, :][index[i, :]]]
                                                   for i in range(temp_features.shape[0])])

                temp_feature_names = np.squeeze(np.array(self.feature_names)[index])
                yield temp_feature_names, temp_features_subset, contributions_subset, temp_preds
        except StopIteration:
            logger.debug('_contrib_reducer_gen completed')

    def _summary_gen(self, feature_names, features, contributions, preds, **kwargs):
        '''

        :param feature_names:
        :param features:
        :param contributions:
        :param preds:
        :param kwargs:
                  'name': False,
                  'date': False,
                  'features': True,
                  'contributions': True,
                  'mean': True,
                  'rank': True,
                  'stdev': True
        :return:
        '''
        from datetime import datetime
        params = {'name': False,
                  'date': False,
                  'features': True,
                  'contributions': True,
                  'mean': True,
                  'rank': True,
                  'stdev': True}
        params.update(*kwargs)
        logger.debug('_summary gen init')
        if all((feature_names.shape != features.shape, features.shape != contributions.shape)):
            logger.error('feature names.txt, contributions and features have miss matched shape in summary gen')
        else:
            logger.debug('passed summary gen shape check')
            logger.debug('summary gen input shape' + str(features.shape))
        n_obs = len(preds)
        try:
            deciles = np.ravel(self.predict_deciles(preds=preds))
        except AttributeError:
            deciles = np.array(['NA'] * n_obs)
            logger.error('decile predict failed inside _summary_gen, mapping to NA')
        logger.debug('summary gen  deciles shape: ' + str(deciles.shape))
        for i in range(n_obs):
            try:
                temp_pred = preds[i]
                temp_features = features[i, :]
                temp_contributions = contributions[i, :]
                temp_feature_names = feature_names[i, :]
            except IndexError:
                ## handels the case where there is only one row of data
                logger.debug('summary gen using one row of data method due to index errors')
                temp_features = features
                temp_contributions = contributions
                temp_feature_names = feature_names
                temp_pred = preds[0]
                logger.debug('summary gen preds: ' + str(preds.shape))
                logger.debug('summary gen feature_shape: ' + str(temp_features.shape))
                logger.debug('summary gen feature_contributions_shape: ' + str(temp_contributions.shape))
                logger.debug('summary gen feature_names: ' + str(temp_feature_names.shape))
            try:
                stdevs = (temp_pred - self.performance['mean']) / self.performance['std']
            except ZeroDivisionError:
                stdevs = "NA"

            temp_decile = deciles[i]
            output = ''
            if params['name']:
                output = output + 'Model Name: ' + self.name + ' \r\n '
            output = output + ' Prediction: ' + '[' + str(round(temp_pred, 3)) + ']'
            if params['rank']:
                output = output + ', Rank: [' + str(temp_decile) + '] \n\r '
            if params['mean']:
                output = output + ' Mean Population Prediction: ' + str(round(self.performance['mean'], 3))
            if params['stdev']:
                output = output + ', Stdevs from Mean: ' + str(round(stdevs, 3)) + ' \r\n '
            if params['date']:
                output = output + ' Date of Prediction: ' + str(datetime.now()) + ' \r\n '
            if params['contributions'] or params['features']:
                output = output + ' Contributing Features: \r\n '
                index = np.argsort(-temp_contributions)
                for _, j in enumerate(index):
                    output = output + '   + ' + str(temp_feature_names[j])
                    if params['features']:
                        output = output + ': value: ' + str(round(temp_features[j], 3))
                    if params['contributions']:
                        output = output + ', estimated contribution: ' + str(round(temp_contributions[j], 3)) + ' \r\n '
            yield output
        logger.debug('summary gen completes in n_iterations: ' + str(n_obs))

    def _standard_output_gen(self, batch_size=10, n=5, **kwargs):
        '''
        internal method called by predict standard output
        :param batch_size: int
        :param n: int number of feature in feature summary
        :param kwargs: kwarges to summary gen
        :return:
        '''
        from datetime import datetime
        time_stamp = datetime.now()
        logger.debug('standard output gen init')
        gen = self._contrib_reducer_gen(n=n, batch_size=batch_size)
        try:
            while True:
                temp_feature_names, temp_features_subset, contributions_subset, temp_preds = next(gen)
                context = list(
                    self._summary_gen(temp_feature_names, temp_features_subset, contributions_subset, temp_preds,
                                      **kwargs))
                rank = np.ravel(self.predict_deciles(preds=temp_preds))
                output_dict = {'Prediction': temp_preds, 'Context': context, 'Rank': rank}
                output = pd.DataFrame(output_dict)
                output['PredictionDate'] = time_stamp
                output['ModelName'] = self.name
                output['ModelInitDate'] = self.model_init_date
                yield output
        except StopIteration:
            logger.debug('standard output gen completed')

    def predict_standard_output(self, data=None, features=None, cols=None, batch_size=100, n=5, **kwargs):
        '''
        primary predict method, using
        can use feature transform and predict method
        Features, if passed in the transform method is not called.


        :param data: pandas data frame
        :param features: numpy array or scipy coo matrix of features (optional)
        :param cols: list of columns to pass through from the data input (generally a primary key)
        :param batch_size: int
        :param n:  int number of feature in feature summary
        :param kwargs: optional kwargs for the summary generator
        :return: pandas data frame
        '''
        logger.debug('predict standard output called')
        self._validate_input(data, features, cols=cols)
        gen = self._standard_output_gen(batch_size=batch_size, n=n, **kwargs)
        output_list = list(gen)
        output = pd.concat(output_list)
        if cols is not None:
            temp_data = col_sub_setter(data, cols)
            temp_data.index = output.index
            output = output.merge(temp_data, left_index=True, right_index=True, how='inner')
        else:
            logger.debug('by passing col selection pass through')
        return output

    def save(self, file_name):
        import pickle
        import os
        logger.debug('saving model to : ', os.getcwd() + '/' + file_name)
        self.features = None
        self.data = None
        self.outputs = None
        pickle.dump(self, open(file_name, 'wb'))


class BagpipeDeploy:
    '''
     A packge to assist in the deployment of multiple bagpipe objects with the same feature transformation step
    '''

    def __init__(self, files=None, bagpipes=None, keep_cols=None, date_col=None, **kwargs):
        '''
        Sets up arguments, loads the file list of bagpipe objects
        :param file_list:
        :param keep_cols:
        :param date_col:
        :param kwargs:
        '''
        self.files = set()
        self.bagpipes = {}
        self.transform_dict = {}
        self.keep_cols = keep_cols
        self.date_cols = date_col
        self.input_cols = set()
        self._params = kwargs
        self.transform_list = None
        self.load_pipes(files, bagpipes)

    def load_pipes(self, files=None, bagpipes=None):
        '''
        loads a list of serialized bagpipe files and checks that the feature names.txt match,
        also set joblib from sklearn to 1
        :param file_list: list of file paths
        :return: list of loaded bag pipe objects
        '''
        import pickle
        logger.debug('load_pipes called')
        pipe_list = []
        if bagpipes is not None:
            if type(bagpipes) is not list:
                bagpipes = [bagpipes]
            pipe_list = bagpipes
        if files is not None:
            if type(files) is str:
                files = [files]
            for f in files:
                logger.debug('loading ' + f)
                pipe = pickle.load(open(f, 'rb'))
                pipe.pipe.n_jobs = 1
                pipe_list.append(pipe)
        for p in pipe_list:
            name = p.name
            self.bagpipes.update({name: p})
        for _, (name, pipe) in enumerate(self.bagpipes.items()):
            self.input_cols.update(pipe.input_names)

    def _data_gen(self, data, batch_size=100):
        from FeaturePipe.utils import batch_gen
        logger.debug('data gen init')
        n_obs = data.shape[0]
        gen = batch_gen(n_obs, batch_size)
        try:
            while True:
                index = next(gen)
                try:
                    output = data.iloc[index, :]
                except AttributeError:
                    logger.debug('bp deploy data gen, using numpy indexing instead of pandas')
                    output = data[index, :]
                yield output
        except StopIteration:
            logger.debug(' data gen completed')

    def _validate_input_data(self, data):
        try:
            missing = self.input_cols.difference(data.columns)
            if missing:
                logger.error('bp deploy missing inputs columns: ' + str(missing))
            else:
                logger.debug('bp deply input columns validated')

        except AttributeError:
            logger.debug(' input column names.txt could not be validated due to data.columns attribute not being populated')

    def _get_pipe_key(self, pipe):
        pass

    def _predict_gen(self, data, batch_size=100):
        path = []
        for _, (name, bp) in enumerate(self.bagpipes.items()):
            path.append((name, bp.get_pipe_key()))
        path.sort(key=lambda tup: tup[1])
        previous_key = ''
        features = None
        gen = self._data_gen(data, batch_size=batch_size)
        try:
            while True:
                temp_data = next(gen)
                for i, tup in enumerate(path):
                    name = tup[0]
                    new_key = tup[1]
                    bp = self.bagpipes[name]
                    if new_key == previous_key:
                        logger.debug('bp deploy reusing feature set')
                        preds = bp.predict_standard_output(features=features)
                        logger.debug('bp deploy predicting using ' + name)
                    else:
                        logger.debug('bp deploy transforming feature set')
                        features = bp.transform(temp_data)
                        preds = bp.predict_standard_output(features=features)
                        logger.debug('bp deploy predicting using ' + name)
                    yield preds
        except StopIteration:
            logger.debug('bp deploy predict gen completed')

    def predict(self, data, batch_size=100):
        logger.debug('bp deploy predict method called')
        ouput_list = self._predict_gen(data, batch_size=batch_size)
        output = pd.concat(ouput_list)
        return output
