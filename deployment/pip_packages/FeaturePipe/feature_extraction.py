from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging
from FeaturePipe import setup_logger
logger = logging.getLogger('FeaturePipe.feature_extraction')


class Text2List(BaseEstimator, TransformerMixin):
    def __init__(self, join_char=' : '):
        self.join_char = join_char
        self.input_type = None
        self.input_shape = None
        self.input_names = set()
        self.attribute_names = None

    def fit(self, X=None, y=None, ):
        self.input_type = type(X)
        if hasattr(X, 'shape'):
            self.input_shape = X.shape
        if hasattr(X, 'columns'):
            self.input_names = set(X.columns)
        self.attribute_names = self.get_feature_names()
        return self

    def get_feature_names(self):
        return list(self.input_names)

    def _data_gen(self, X):
        logger.debug('text 2 list transform gen called')
        try:
            n_obs = X.shape[0]
            for i in range(n_obs):
                try:
                    try:
                        yield X.iloc[i, :]
                    except IndexError:
                        yield X.iloc[i]
                except AttributeError:
                    try:
                        yield X[i, :]
                    except IndexError:
                        yield X[i]
        except AttributeError:
            logger.debug('text to list using  list generator')
            n_obs = len(X)
            for i in range(n_obs):
                yield X[i]

    def _transform_gen(self, X):
        try:
             missing = self.input_names.difference(set(X.columns))
             if missing:
                 logger.error('texts to list input colunams failed vailidion')
             else:
                logger.debug('text to list input cols validated')
        except AttributeError:
            logger.debug('text to list input cols could not validated')

        gen = self._data_gen(X)
        try:
            while True:
                temp_data = next(gen)
                if type(temp_data) is str:
                    output = temp_data
                elif hasattr(temp_data, '__iter__') is False:
                    output = str(temp_data)
                elif temp_data is None:
                    output = ''
                else:
                    temp_data = [str(t)if t is not None else '' for t in temp_data]
                    output = self.join_char.join(temp_data)
                yield output
        except StopIteration:
            logger.debug('text to list transform gen completed')

    def transform(self, X):
        gen = self._transform_gen(X)
        output = list(gen)
        return output


class TextFeature(BaseEstimator, TransformerMixin):
    '''
    A class to fit and transform text data
    Uses a helper class from TMM pipe Text2List to normalize input, so the
    count vectorizer or keras tokenizer always get a string list input (with no integers or None)

    Can be called inside inheriets BaseEstimator, TransformerMixin  so it can sklearn pipeline
    Usage:

    if __name__ == "__main__":
        text = fetch_20newsgroups().data
        data = pd.DataFrame({'x': text[1:] + [None], 'y': text[:-1] + [1]})
        n = 100
        t = TextFeature(back_end='keras')
        t.fit(data)
        features = t.transform(data)
    '''

    def __init__(self, back_end='sklearn', clean=True, clean_params=None, par=False, split=' ', **kwargs):
        '''
        defaults to 100 features, max_features (sklearn) or num_words (keras) are not specified
        :param back_end: 'sklearn' (default) or 'keras'
        :param clean: logical apply cleaning function (rm_numbers, stop_words.txt, emails, punctuation)
        :param clean_params: optional dictionary of parameters to pass into word.preprocessing.py.Cleaner class
        :param verbose: logical to print output
        :param kwargs: kwarges argmuments to sklearn count vectorizer, or keras tokenizer
        '''
        from words import Cleaner
        import warnings
        warnings.simplefilter('ignore', category=FutureWarning)
        logger.debug('Text Feature init called')
        logger.debug('Text Feature backend: ' + back_end)
        self.back_end = back_end
        self.clean = clean
        self.par = par
        self.feature_names = None
        self.split = split
        self.use_splitter = False
        if self.split != ' ':
            self.clean = False
            self.use_splitter = True

        if back_end == 'keras':
            from keras.preprocessing.text import Tokenizer
            if 'num_words' not in kwargs.keys():
                self.tokenizer = Tokenizer(num_words=100, split=self.split,  **kwargs)
            else:
                self.tokenizer = Tokenizer(**kwargs)
            self.text2list = Text2List(join_char=self.tokenizer.split)
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            if 'max_features' not in kwargs.keys():
                self.count_vect = CountVectorizer(max_features=100,  **kwargs)
            else:
                self.count_vect = CountVectorizer(**kwargs)
            self.text2list = Text2List()
        self.cleaner = Cleaner(params=clean_params, par=par)

    def fit(self, data, verbose=False):
        '''
        Fit text encoders on a list,data frame or array of data
        :param data: pondas data frame, list or array (2d data with be joined with a seperation character ' ' by default
        :param verbose: logical print debugging output
        :return: fitted TextFeature Object
        '''
        self.text2list.fit(data)
        data = self.text2list.transform(data)
        if self.clean:
            logger.debug('cleaning ...')
            data = self.cleaner.fit_transform(data)
        if self.use_splitter:
            logger.debug('split: ' + self.split + ', custom splitting  ... ')
            data = self._splitter(data)
            logger.debug('back_end :' + self.back_end +' fitting ...')
        if self.back_end == 'keras':
            self.tokenizer.fit_on_texts(data)
            n_words = len(self.tokenizer.word_index.keys())
            self.tokenizer.num_words = min(n_words, self.tokenizer.num_words)
        else:
            self.count_vect.fit(data)
        self.feature_names = self.get_feature_names()
        return self

    def _transform_gen(self, data, clean=True, text_2_list=True):
        '''
        row wise feature extraction generator based on the fitted tokenizer or count vectorizer
        :param data: pondas data frame, list or array
        :return: scipy coo matrix of features
        '''
        from scipy.sparse import coo_matrix
        if text_2_list:
            data_list = self.text2list.transform(data)
        else:
            if type(data) is list:
                data_list = data
            else:
                data_list = list(data)
        for i, d in enumerate(data_list):
            if clean:
                logger.debug('cleaning ... ')
                d = self.cleaner.transform(d)
                logger.debug('text to matrix with back_end: ', self.back_end)
            if self.back_end == 'keras':
                yield coo_matrix(self.tokenizer.texts_to_matrix([d]))
            else:
                yield self.count_vect.transform([d])

    def transform(self, data, verbose=False):
        '''
        primary feature extraction method on a list, data frame or array of text using fitted tokenizer or count vect
        :param data: pondas data frame, list or array
        :return: scipy coo matrix of features
        '''
        from scipy.sparse import vstack
        data = self.text2list.transform(data)
        if self.clean:
            if verbose:
                logger.debug('cleaning ...')
            data = self.cleaner.transform(data)
        if verbose:
            logger.debug('transforming with back_end: ', self.back_end)
        feature_list = list(self._transform_gen(data, clean=False, text_2_list=False))
        if verbose:
            logger.debug('extaction completed, vstacking')
        features = vstack(feature_list)

        return features

    def get_feature_names(self):
        if self.back_end == 'keras':
            feature_names = list(self.tokenizer.word_index.keys())
            n = self.tokenizer.num_words
            if n is not None:
                feature_names = feature_names[0:n]
        else:
            feature_names = self.count_vect.get_feature_names()
        return feature_names

    def _splitter(self, data):
        return list(self.__split_gen(data))

    def __split_gen(self, data):
        return_list = False
        for d in data:
            if type(d) is list:
                d = d[0]
                return_list = True
            d_split = d.split(self.split)
            d_split = [k.strip() for j, k in enumerate(d_split)]
            d_joined = self.split.join(d_split)
            yield d_split


class Decilizer:
    '''
    A class to calculate deciles on data frames or arrays
    Works on lists, pandas data frames and numpy arrays
    based on a uniform distrobution
    uses KBinsDiscritizer under the hood/

    usage:
    x = np.arange(100)/50
    d = Decilizer()
    d.fit(x)
    d.transform(x)

    '''

    def __init__(self, n=100, **kwargs):
        self.n = n
        self.params = {'encode': 'ordinal', 'strategy': 'quantile'}
        self.n_cols = 1
        self.params.update(**kwargs)
        self.deciles = None

    def input(self, data):
        if hasattr(data, 'shape'):
            if len(data.shape) == 1:
                logger.debug('decilier reshaped data to (-1, 1')
                data = np.reshape(data, (-1, 1))
            else:
                logger.debug('deciler maintining original data shape: ' + str(data.shape))
        else:
            data = np.reshape(data, (-1, 1))
            logger.debug('decilier reshaped data to (-1, 1')
        return data

    def fit(self, data):
        '''
        takes a 1d array of data, bins it into integer deciles from 1-100 by default

        :param data: numpy array of 1d, or shape (n, 1)
        :param n: number of decile bins
        :param kwargs sklearn KBinsDiscritizer encode parameter
        :return: none
        '''
        import numpy as np
        from sklearn.preprocessing import KBinsDiscretizer
        d = KBinsDiscretizer(n_bins=self.n, **self.params)
        data = self.input(data)
        d.fit(data)
        self.deciles = d

    def transform(self, data):
        '''
        transforms an array into bins
        :param data: numpy array of 1d, or shape (n, 1)
        :return: array of integer bins 1:100 by default
        '''

        import numpy as np
        from itertools import chain
        data = self.input(data)
        deciles = self.deciles.transform(data)
        return np.array(deciles, dtype=np.int) + 1
