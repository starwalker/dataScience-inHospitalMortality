import logging
from FeaturePipe import setup_logger
logger = logging.getLogger('FeaturePipe.adapter')

class Adapter:
    def __init__(self):
        '''
        Adapter is intended to adapt one data frame to look like another, useful when training data was different
        than the production environment
        Author: Julian Smith

        a = Adapter()
        a.blacklist = ['PatientID', 'CSN'] # columns to ignore ( and pass through un altered)
        a.add_scrub('BloodPressure', r'([0-9]+)(?=\/[0-9]+)', 'BLOOD PRESSURE (SYSTOLIC)')   # scrub out no numeric data
        a.add_rule('GLUCOSE, WHOLE BLOOD', 'lambda x: 0', 'GLUCOSE') # apply a lambda function to data
        col_map = { 'input_col_name_1': ('output_col_name_1', 'float'),
                    'input_col_name_2': ('output_col_name_2', 'float'),
        a.fit(data)
        a.transform(data)
        '''
        self.col_scrub_dict = {}
        self.col_rule_dict = {}
        self.col_map_dict = {}
        self.col_set_dict = {}
        self.input_cols = None
        self.output_cols = None
        self.blacklist = set()

    def apply_scrub(self, data, col, pattern=None, new_col=None, dtype=None):
        """
        Applies regex to column of input_data
        :param col: a column in input_data
        :return: void
        """
        from pandas import to_numeric
        import re

        if new_col is None:
            # no new column is created
            new_col = col

        if dtype is None:
            dtype = 'float'

        try:
            data[new_col] = to_numeric(data[col])

        except ValueError:
            if pattern is not None:
                try:
                    data[new_col] = data[col].str.extract(pattern, expand=False)
                    data[new_col] = data[new_col].astype(dtype)
                except TypeError:
                    raise TypeError('Regex {0} applied to col {1} is not valid'.format(pattern, col))
                except ValueError:
                    raise ValueError('Cannot convert col {0} to dtype {1}'.format(col, dtype))
            else:
                try:
                    # logic for extracting the first possible numeric by row
                    data[new_col] = data[col].apply(
                        lambda x: re.findall(r'(-?\d+\.?\d*)', str(x))[0] if len(
                            re.findall(r'(-?\d+\.?\d*)', str(x))) > 0 else None).astype(dtype)
                except TypeError:
                    raise TypeError('Regex {0} applied to col {1} is not valid'.format(pattern, col))
                except ValueError:
                    raise ValueError('Cannot convert col {0} to dtype {1}'.format(col, dtype))

        return data

    def add_scrub(self, col, pattern=None, new_col=None, dtype='float'):
        """
        Add scrub schema to col_scrub_dict
        :param col: a string, col to scrub
        :param pattern: a regex string
        :param new_col: a string, col to rename, if none don't rename
        :param dtype: a dtype to convert the scrubbed product to
        :return: void
        """
        if col in self.col_scrub_dict.keys():
            # there may be multiple scrubs of the same column (ex: Blood Pressure --> [BP (SYSTOLIC), BP (DIASTOLIC)])
            self.col_scrub_dict[col].append((pattern, new_col, dtype))
        else:
            self.col_scrub_dict[col] = [(pattern, new_col, dtype)]

    def scrub_transform(self, data):
        """
        Performs a scrub transform on the input dataframe
        :return: void
        """
        for col in data.columns:
            if col not in self.blacklist:
                if col in self.col_scrub_dict.keys():
                    for item in self.col_scrub_dict[col]:
                        # handle multiple scrubs on one column
                        pattern = item[0]
                        new_col = item[1]
                        dtype = item[2]
                        data = self.apply_scrub(data, col, pattern, new_col, dtype)
                else:
                    data = self.apply_scrub(data, col)

        return data

    def add_rule(self, col, func, col_from, dtype='float'):
        """
        :param col: the col name for the new rule-derived col
        :param func: a lambda function
        :param col_from: a col name in input cols from which the flags are pulled
        :param dtype: a string dtype
        :return:
        """
        self.col_rule_dict[col] = (func, col_from, dtype)

    def rule_transform(self, data):
        """
        Performs the rule transform on the input dataframe
        :return: void
        """
        for col, values in self.col_rule_dict.items():
            func = values[0]
            col_from = values[1]
            dtype = values[2]
            try:
                if isinstance(eval(func), type(lambda: 0)):
                    func = eval(func)
            except ValueError:
                raise ValueError('rules funcs must be lambda/lambda is not valid')
            data[col] = data[col_from].astype(dtype).apply(func)
        return data

    def add_map(self, col, new_col, dtype='float'):
        """
        Adds a mapping to the col map dict
        :param col: a col in input cols
        :param new_col: a col string to rename col to
        :param dtype: dtype for col mapping
        :return:
        """
        self.col_map_dict[col] = (new_col, dtype)

    def map_transform(self, data):
        for col, values in self.col_map_dict.items():
            new_col = values[0]
            dtype = values[1]

            data.rename(columns={col: new_col}, inplace=True)
        return data

    def add_set(self, col, value):
        self.col_set_dict[col] = value

    def set_transform(self, data):
        for col, value in self.col_set_dict.items():
            data[col] = value
        return data

    def fit(self, data):
        '''
        validate transforms
        :return: void
        '''
        data = data.copy()
        sample_data = data.head(2)
        self.input_cols = set(data.columns)
        try:
            data = self.scrub_transform(data)
            logger.debug('scrub transform passed.')
        except ValueError:
            raise ValueError('Error in scrub transform')
        try:
            data = self.map_transform(data)
            logger.debug('map transform passed.')
        except ValueError:
            raise ValueError('Error in map transform')
        try:
            data = self.rule_transform(data)
            logger.debug('rule transform passed.')
        except ValueError:
            raise ValueError('Error in rule transform')
        try:
            data = self.set_transform(data)
            logger.debug('set transform passed.')
        except ValueError:
            raise ValueError('Error in set transform')
        new_data = self.transform(sample_data)
        self.output_cols = set(new_data.columns)

    def transform(self, data):
        '''
        Applies all the col_maps and the functions to input data
        :return:
        '''
        from pandas import DataFrame
        if not isinstance(data, DataFrame):
            raise ValueError('data must be a pandas dataframe')
        data = data.copy()
        missing = set.difference(self.input_cols, set(data.columns))
        if missing:
            raise KeyError('missing columns: {0} from data'.format(missing))
        data = self.scrub_transform(data)
        data = self.map_transform(data)
        data = self.rule_transform(data)
        data = self.set_transform(data)
        return data

    def propose_col_mapping(self, input_cols, output_cols):
        """
        uses knn to suggest possible column mappings
        :return: col_map
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.neighbors.classification import KNeighborsClassifier

        if len(output_cols) == 0:
            raise ValueError('No columns in output cols specified')

        col_map = {}
        cv = CountVectorizer(analyzer='char')
        neigh = KNeighborsClassifier(n_neighbors=1)
        cv_cols = cv.fit_transform(output_cols)
        neigh.fit(cv_cols.todense(), output_cols)
        for col in input_cols:
            if col not in output_cols and col not in self.blacklist:
                cv_outcol = cv.transform([col])
                col_map[col] = (''.join(neigh.predict(cv_outcol.todense())), 'float')

        return col_map
