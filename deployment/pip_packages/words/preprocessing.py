from sklearn.base import BaseEstimator, TransformerMixin
import re
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from FeaturePipe import setup_logger
logger = logging.getLogger('words.feature_extraction')

class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        '''

        :param kwargs: key word arguments passed into the  clean_strings function
        '''
        logger.debug('Cleaner class init')
        self.params = {'split_char': ' ',
                       'rm_numbers': True,
                       'rm_symbols': True,
                       'rm_emails': True,
                       'rm_stop_words': True,
                       'replace_><': False,
                       'replace_abbreviations': True,
                       'lower': True,
                       'stem': False,
                       'stop_words': 'all',
                       'stemmer': None,
                       'abbreviations': None,
                       'par': False,
                       }
        self._update(**kwargs)

    def _update(self, params=None, **kwargs):
        '''
        internal update method
        :param params: params dict  passed into the  clean_strings function
        :param kwargs: kwargs: key word arguments passed into the  clean_strings function
        :return:
        '''
        logger.debug('update method called')
        if params is not None:
            for key in params.keys():
                if key in self.params.keys():
                    self.params[key] = params[key]
                else:
                    raise KeyError(key + ' is an unknown keyword argument')

        # handles parameters entered as key words
        for key in kwargs.keys():
            if key in self.params.keys():
                self.params[key] = kwargs[key]
            else:
                raise KeyError(key + ' is an unknown keyword argument')

        # setup the stemmer
        if all((self.params['stem'], self.params['stemmer'] is None)):
            from nltk.stem import SnowballStemmer
            self.params['stemmer'] = SnowballStemmer("english")

        # load the stop word list
        if self.params['rm_stop_words']:
            if type(self.params['stop_words']) is not list:
                if self.params['stop_words'] in ['english', 'names.txt', 'musc']:
                    self.params['stop_words'] = set(load_stop_words(self.params['stop_words']))
                else:
                    self.params['stop_words'] = load_stop_words('all')
            if type(self.params['stop_words']) is list:
                self.params['stop_words'] = set(self.params['stop_words'])
        else:
            self.params['stop_words'] = None

        # load medidcal abreviation replaceer
        if self.params['replace_abbreviations']:
            self.params['abbreviations'] = load_abbreviations()
        else:
            self.params['abbreviations'] = None
        logger.debug('updated with new params' + str(params))

    def _clean_docs_gen(self, docs):
        '''

        '''
        params = self.params
        if type(docs) is str:
            docs = [docs]
        for doc in docs:
            # replaces <> with greaterthan lessthan
            if params['replace_><']:
                doc = re.sub('[>]', ' greater ', doc).strip()
                doc = re.sub('[<]', ' less ', doc).strip()
                logger.debug('>< replace: ' + str(doc))

            # remove numbers
            if params['rm_numbers']:
                doc = re.sub('\d', ' ', doc)
                logger.debug('rm nums: ' + str(doc))

            # remove symbols
            if params['rm_symbols']:
                doc = re.sub('[\W_]+', ' ', doc)
                logger.debug('rm symbols ' + str(doc))

            # lower case everything
            if params['lower']:
                doc = doc.lower()

            # tokenize document
            tokens = doc.split()
            logger.debug('tokenized: ' + str(tokens))
            tokens_cleaned = list(self._clean_toke_gen(tokens))
            tokens_joined = self.params['split_char'].join(tokens_cleaned)
            yield tokens_joined.strip()

    def _clean_toke_gen(self, tokens):
            import re
            params = self.params
            stop_words = self.params['stop_words']
            med_lookup_dict = self.params['abbreviations']
            stemmer = self.params['stemmer']
            for token in tokens:
                output_token = token
                # remove emails
                if params['rm_emails']:
                        if re.search("[^\s]+@[^\s]+[.][^\s]{3}", token):
                            output_token = None
                            logger.debug('email removed: ' + str(token))
                # remove stop words
                if all((output_token is not None, self.params['stop_words'] is not None)):
                    if token not in stop_words:
                        output_token = token
                    else:
                        output_token = None
                        logger.debug('stop words removed: ' + str(output_token))
                # look up medical abbreviations
                if all((output_token is not None, med_lookup_dict is not None)):
                    try:
                        output_token = med_lookup_dict[token]
                    except KeyError:
                        output_token = token
                        logger.debug('abbreviations replaced: ' + str(output_token))
                # stem
                if all((output_token is not None, stemmer is not None)):
                    output_token = stemmer.stem(output_token)
                    logger.debug('stemmed: ' + str(output_token))
                if output_token is None:
                    output_token = ''
                yield output_token

    def _clean_doc_helper(self, doc):
        return list(self._clean_docs_gen(doc))[0]

    def fit(self, X, y=None, **kwargs):
        self._update(**kwargs)
        return self

    def transform(self, X):
        from time import time
        logger.info('Cleaner Transform method called')
        start = time()
        if type(X) is str:
            data = [X]
        else:
            data = list(map(str, X))

        logger.debug('data in: ' + str(data))
        if hasattr(data,  '__iter__'):
            n_docs = len(data)
            if self.params['par']:
                from multiprocessing import Pool, cpu_count
                n_workers = min(n_docs, cpu_count())
                logger.debug('running in parallel with n_workers:' + str(n_workers) + ' n_docs: ' + str(n_docs))
                p = Pool(n_workers)
                output = p.map(self._clean_doc_helper, data)
                p.close()
                p.join()
                p.terminate()
            else:
                logger.debug('running on generators with n_docs: ' + str(n_docs))
                output = list(self._clean_docs_gen(data))
            total = time()-start
            try:
                logger.debug('docs per second: ' + str(n_docs/total) + 'total time: ' + str(total))
            except ZeroDivisionError:
                pass
            return output
        else:
            raise ValueError('data is not iterable, recieved: ' + str(type(data)))


def load_abbreviations():
    import pkg_resources
    logger.info('loading abbreviations ...')
    word_dict = {}
    path = 'resources//med_abbv.txt'
    stream = pkg_resources.resource_stream('words', path)
    for i, bytes in enumerate(stream):
        new_word = bytes.decode(errors='ignore')
        new_word = new_word.replace('\r', '')
        new_word = new_word.replace('\n', '')
        new_word = new_word.replace('\t', '')
        new_word = new_word.replace('\xa0', '')
        new_word = new_word.strip()
        new_word = new_word.lower()
        new_word = new_word.split(',')
        word_dict[new_word[0]] = new_word[1]
    return word_dict


def load_stop_words(word_list='all', lower=True):
    '''
        * stopwords 'english' is references here
            https://www.ranks.nl/stopwords
        * musc is a comon list of medical stop words
        * all loads load stop words
        * names.txt is the top list of 200 first names.txt

    :param word_list: str: in 'all', 'english', 'names.txt', 'musc'
    :param lower: logical, return lower case
    :return: a list of stop words
    '''
    import pkg_resources
    logger.info('loading stop words ...')
    dir = 'resources/'
    names_path = dir + 'names.txt'
    stops_path = dir + 'stop_words.txt'
    musc_stops_path = dir + 'musc_stop_words.txt'
    if word_list in ['all', True, None, '']:
        file_path_list = [names_path, stops_path, musc_stops_path]
    elif word_list in ['english', 'stopwords', 'stop_words']:
        file_path_list = [stops_path]
    elif word_list in ['names.txt', 'name_list', 'first_names']:
        file_path_list = [names_path]
    elif word_list in ['musc', 'MUSC']:
        file_path_list = [musc_stops_path]
    else:
        raise ValueError(word_list + ' not in known lists of words')
    words = []
    for path in file_path_list:
        stream = pkg_resources.resource_stream('words', path)
        for i, bytes in enumerate(stream):
            new_word = bytes.decode("utf-8")
            new_word = new_word.replace('\r', '')
            new_word = new_word.replace('\n', '')
            new_word = new_word.replace(',', '')
            words.append(new_word)
    words = list(set(words))
    if lower:
        words = list(map(lambda x: x.lower(), words))
    return words


def abbreviation_replacer(word_list, look_up_dict):
    logger.debug('replacing abreviations')
    if type(word_list) is str:
        word_list = [word_list]
    output = [look_up_dict[w] if w in look_up_dict.keys() else w for i, w in enumerate(word_list)]
    return output
