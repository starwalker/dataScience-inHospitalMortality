import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from collections import Counter
import numpy as np
from scipy.sparse import csc_matrix, vstack
import xgboost as xgb
import logging
from FeaturePipe import setup_logger
logger = logging.getLogger('FeaturePipe.utils')
logger.setLevel(logging.WARNING)


def make_names(data):
    '''

    :param data: a str or list of strings
    :return: clean list of strings for use in naming xgb boost columns
    '''
    import re
    if type(data) is str:
        data = [data]
    data = list(map(str, data))
    def clean_string(x):
        x = re.sub('>', ' lessthan ',  x)
        x = re.sub('<', ' greaterthan ', x)
        x = re.sub('[/&]', ' and ', x)
        x = re.sub('%', 'percent', x)
        x = re.sub('[\\[\\]\\{\\}\\?\\!\\,\\:\\;\\@\\^\\%]', ' ', x)
        return x
    return list(map(clean_string, data))


def class_performance(labels, preds, cut_off=.5, fill=None, ref=1):
    '''
    usage:
        labels =  np.random.choice([0,1], 100)
        preds = np.random.rand(100)
        perf = class_performance(labels, preds, .5)

    :param labels: list or array or list of 0/1 true outcomes
    :param preds: list or array of floating probabilities or 0/1 outcomes
    :param cut_off: float (probablity cut off)
    :param ref: int or str, reference True class (defaulted to 1)
    :param fill: int, str, None value to fill when there is a zero division, or the measure is not calculable.
    :return: dictionary of performance measures with keys
    'tp', 'fp', 'fn', 'tn',
    'tpr', 'tnr', 'fpr', 'fdr', 'For',
    'acc', 'f1', 'mcc', 'auc',
    'mean_labels', 'mean_preds', 'mean_prob'
    '''
    import numpy as np
    from math import sqrt
    import warnings
    from sklearn.metrics import roc_auc_score
    n_labels = len(labels)
    n_preds = len(preds)
    if n_labels != n_preds:
        raise ValueError('n_labels:' + str(n_labels) + ' != to n_preds: ' + str(n_preds))
    output = dict()
    output['n_obs'] = n_labels
    output['mean_prob'] = np.mean(preds)
    output['mean_labels'] = np.mean(labels)
    output['no_info_accuracy'] = 1 - output['mean_labels']

    if len(np.unique(preds) > 2):
        output['auc'] = roc_auc_score(labels, preds)
        preds = [1 if p > cut_off else 0 for p in preds]
    else:
        warnings.warn('assuming that preds are an actual class prediction, no aprobabilityy n.unique =< 2')
        output['auc'] = fill
    output['mean_pred'] = np.mean(preds)
    try:
        preds = np.array(preds, dtype=np.int)
        labels = np.array(labels, dtype=np.int)
    except ValueError:
        warnings.warn('preds and labels could not be coerced to integer arrays')

    tp = sum([1 if all((j[0] == ref, j[1] == ref)) else 0 for i, j in enumerate(zip(labels, preds))])
    fp = sum([1 if all((j[0] != ref, j[1] == ref)) else 0 for i, j in enumerate(zip(labels, preds))])
    tn = sum([1 if all((j[0] != ref, j[1] != ref)) else 0 for i, j in enumerate(zip(labels, preds))])
    fn = sum([1 if all((j[0] == ref, j[1] != ref)) else 0 for i, j in enumerate(zip(labels, preds))])
    output['tp'] = tp
    output['fp'] = fp
    output['tn'] = tn
    output['fn'] = fn

    try:
        output['tpr'] = tp / (tp + fn)
    except ZeroDivisionError:
        output['tpr'] = fill
    try:
        output['tnr'] = tn / (tn + fp)
    except ZeroDivisionError:
        output['tnr'] = fill
    try:
        output['ppv'] = tp / (tp + fp)
    except ZeroDivisionError:
        output['ppv'] = fill
    try:
        output['npv'] = tn / (tn + fn)
    except ZeroDivisionError:
        output['npv'] = fill
    try:
        output['fnr'] = fn / (fn + tp)
    except ZeroDivisionError:
        output['fnr'] = fill
    try:
        output['fpr'] = 1 - output['tnr']
    except TypeError:
        output['fpr'] = fill
    try:
        output['fdr'] = 1 - output['ppv']
    except TypeError:
        output['fdr'] = fill
    try:
        output['For'] = 1 - output['npv']
    except TypeError:
        output['For'] = fill
    try:
        output['acc'] = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        output['acc'] = fill
    try:
        output['f1'] = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        output['f1'] = fill
    try:
        output['mcc'] = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except ZeroDivisionError:
        output['mcc'] = fill
    return output


def optimum_cutoff(labels, preds, key='acc', cut_range=(.01, .99), step=.01, maximize=True):
    '''
    usage:
        labels =  np.random.choice([0,1], 100)
        preds = np.random.rand(100)
        cutoff = optimum_cutoff(labels, preds)['best_cutoff']
    :param labels: list or array or list of 0/1 true outcomes
    :param preds: list or array of floating probabilities or 0/1 outcomes
    :param key: str key from class_performance
    :param cut_range: tuple with values between 0-1, defines the range of the cutoff to try
    :param step: float (steps between the ends of the cutoff range
    :param maximize: logical (maximize or minimize the key
    :return: dict including optimim cut off

    '''
    import numpy as np
    n_labels = len(labels)
    n_preds = len(preds)
    if n_labels != n_preds:
        raise ValueError('n_labels:' + str(n_labels) + ' != to n_preds: ' + str(n_preds))
    lower_range = max(min(preds), cut_range[0])
    upper_range = min(max(preds), cut_range[1])
    cutoffs = np.arange(lower_range, upper_range, step)
    performance = np.zeros(len(cutoffs))
    if maximize:
        fill = 0
    else:
        fill = 1
    for i, c in enumerate(cutoffs):
        performance[i] = class_performance(labels, preds, cut_off=c)[key]
    if maximize:
        performance[np.isnan(performance)] = 0
        best_index = np.argmax(performance)
    else:
        performance[np.isnan(performance)] = 1
        best_index = np.argmin(performance)
    best_cutoff = cutoffs[best_index]
    performance_dict = class_performance(labels, preds, cut_off=best_cutoff, fill=fill)
    output = {'cut_offs': cutoffs, 'performance': performance, 'optimized': key, 'best_cutoff': best_cutoff,
              'all_measures': performance_dict}
    return output


def batch_gen(n, batch_size=1):
    '''
    Generate batches of indecies
    usage:
    g = batch_gen(10, 3)
    list(g)
     yeilds: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    g = batch_gen(10, 30)
    list(g)
     yeilds: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    :param n: int (total number of obs)
    :param batch_size: int (size of each batch of indicies
    :return: gen

    '''
    def make_gen(n=n):
        for i in range(n):
            yield i

    if batch_size >= n:
        yield list(range(n))
    else:
        j = 0
        gen = make_gen(n)
        while j < n:
            output = []
            for _ in range(batch_size):
                try:
                    v = next(gen)
                    output.append(v)
                    j += 1
                except StopIteration:
                    pass
            yield output


def load_pipes(file_list):
    '''
    loads a list of serialized bagpipe files and checks that the feature names.txt match,
    also set joblib from sklearn to 1
    :param file_list: list of file paths
    :return: list of loaded bag pipe objects
    '''
    import pickle
    import warnings
    print('loading bagpipe pipelines ...{0}.format(file_list')
    if file_list is str:
        file_list = [file_list]
    output_list = []
    test_list = []
    for i, f in enumerate(file_list):
        pipe = pickle.load(open(f, 'rb'))
        pipe.pipe.n_jobs = 1
        output_list.append(pipe)
        if i > 0:
            test_list.append(output_list[i-1].feature_names == output_list[i].feature_names)
    if len(test_list) == 0:
        print('pipe feature names.txt match')
    elif all((test_list)):
        print('pipe feature names.txt match')
    else:
        warnings.warn('pipe feature name missmatch')
    return output_list


def get_performance(true, preds, cut_off=.5, target_val=1, type=None):
    '''

    :param true: 1d array or list of 0/1 outcomes
    :param preds: 1d array or list of prediction probabilities
    :param cut_off: cut off for the predicted probablities
    :param target_val: int target val in the true list to consider success
    :param type: str key from the list
    'precision', 'recall', 'sensitivity', 'tpr', 'tnr', 'f', 'specificity', 'pprc', 'fpr', 'fnr', 'ppv', 'acc',
    'accuracy', 'dor', 'for', 'fallout', 'fdr', 'tp', 'fp', 'fn', 'tn'
    if specified, returns only the specified value from the dict.
    :return: a dict of performance values
    '''
    from numpy import unique
    if len(preds) != len(true):
        return 'len missmatch between true and preds'
    n_obs = len(true)
    if all((len(unique(preds)) == 2, len(unique(true)) == 2)):
        pred_class = preds
    else:
        pred_class = list(map(lambda x: x > cut_off, preds))
    true_class = list(map(lambda x: x == target_val, true))
    results = list(zip(true_class, pred_class))
    tp = sum(list(map(lambda x: all((x[0] == True, x[1] == True)), results)))
    fp = sum(list(map(lambda x: all((x[0] == False, x[1] == True)), results)))
    fn = sum(list(map(lambda x: all((x[0] == True, x[1] == False)), results)))
    tn = sum(list(map(lambda x: all((x[0] == False, x[1] == False)), results)))
    total = tp + fp + fn + tn
    if total != n_obs:
        print('warning sum of TP, FP TN FN does not = n_obs')
    if tp + fp > 0:
        precision = tp / (tp + fp)
        fdr = fp / (fp + tp)
    else:
        precision = 0
        fdr = 0
    ppv = precision
    if tp + fn > 0:
        recall = tp / (tp + fn)
        fnr = fn / (fn + tp)
    else:
        recall = 0
        fnr = 0
    sensitivity = recall
    tpr = recall
    if tn + fp > 0:
        tnr = tn / (tn + fp)
    else:
        tnr = 0
    specificity = tnr
    pprc = (tp + fp) / (tp + fp + fn + tn)
    if precision + recall > 0:
        f = 2 * (precision * recall) / (precision + recall)
    else:
        f = 0
    if fp + tn > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0
    fallout = fpr
    if fn + tn > 0:
        FOR = fn / (fn + tn)
    else:
        FOR = 0
    acc = (tn + tp)/(tp + fp + fn + tn)
    accuracy = acc
    if all([fpr > 0, fnr > 0, tnr > 0]):
        dor = (tpr/fpr)/(fnr/tnr)
    else:
        dor = 0
    output_dict={'precision': precision, 'recall': recall, 'sensitivity': sensitivity, 'tpr': tpr, 'tnr': tnr, 'f': f,
               'specificity': specificity, 'pprc': pprc, 'fpr': fpr, 'fnr': fnr, 'ppv': ppv, 'acc': acc,
               'accuracy': accuracy, 'dor': dor, 'for':FOR, 'fallout':fallout, 'fdr':fdr, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}
    if type is not None:
        if type in output_dict.keys():
            return output_dict[type]
        else:
            print('value not in keys, keys availible are: ', output_dict.keys())
    else:
        return output_dict


class Keys:
    '''
    A class to create a key out of multiple columns of a pandas data frame

    usge:
        import pandas as pd
        data = pd.DataFrame({'x':[1,2,3], 'y':['a', 'b', 'c']})
        k = Keys(['x', 'y'], data=data)
        keys = list(k.key_gen(data))
        z = pd.concat(list(k.unkey_gen(keys)))
    '''
    def __init__(self, cols, data=None, sep_char=' : '):
        '''

        :param cols: list of string column names.txt
        :param data: pandas data frame including cols in data.columns
        :param sep_char: ' : " string, character that is used to split keys into separate components
        '''
        self.cols = cols
        self.sep_char = sep_char
        self.dtypes = None
        if data is not None:
            self.key_gen(data)

    def key_gen(self, data):
        '''
        Yields a key based on joining as data[self.cols] for each row
        :param data: pandas data frame the includes self.cols column names.txt
        :return: generator
        '''
        from warnings import warn
        from re import findall
        cols = self.cols
        sep_char = self.sep_char
        missing = set(cols).difference(set(data.columns))
        if missing:
            raise KeyError('missing: {0} in input data'.format(missing))
        else:
            if self.dtypes is None:
                self.dtypes = data.dtypes.to_dict()
            for _, row in data.loc[:, cols].iterrows():
                output = sep_char.join(row.astype('str').tolist())
                n_matches = len(findall(sep_char, output))
                if n_matches != len(cols) - 1:
                    warn('n_matches: {0} !=  n_cols - 1: {1}'.format(n_matches, len(cols) - 1))
                yield output

    def unkey_gen(self, x):
        '''
        usage :
            pd.concat(list(k.unkey_gen(keys)))
        yields a pandas data frame for each value in the key list
        :param data: a list of keys
        :return: generator
        '''
        import pandas as pd
        from warnings import warn
        from re import findall
        sep_char = self.sep_char
        cols = self.cols
        dtypes = self.dtypes
        for i, key in enumerate(x):
            n_matches = len(findall(sep_char, key))
            if n_matches != len(cols) - 1:
                warn('n_matches: {0} !=  n_cols - 1: {1}'.format(n_matches, len(cols) - 1))
            keys_split = key.split(sep_char)
            if len(keys_split) != len(cols):
                warn('num cols: {0} and length of key split: {1} are miss matched'.format(len(cols), len(keys_split)))
            output = pd.DataFrame(dict(zip(cols, keys_split)),  index=[i])
            for _, col in enumerate(dtypes.keys()):
                output[col] = output[col].astype(dtypes[col], errors='ignore')
            yield output


def any_none(tup):
    '''
    :param tup: tuple of lists
    :return: logical whether nested lists contain nones
    '''
    def any_none_gen(tup):
        for _, val in enumerate(tup):
            yield any([True if i is None else False for i in val])
    return any(list(any_none_gen(tup)))


def get_lagged_array(x, n_past=0, n_ahead=0, fill=False, fill_val=None):
    '''
        x = np.arange(10)
        get_lagged_array(x ,2, 3)
        get_lagged_array(x ,2, 3, fill=True)
        get_lagged_array(x ,0, 3, fill=True)

    :param x: list or 1d array of sequential data
    :param n_past:int number of past lags
    :param n_ahead:int number of ahead lags
    :return: a generator that yields n_past, n_ahead, index arrays (if n_past, n_ahead >0)
    '''
    from warnings import warn
    import numpy as np
    x = np.squeeze(x)
    gen = n_ahead_gen(x, n_past, n_ahead, fill_val)
    tuple_list = list(gen)
    if len(x) < n_past + n_ahead:
        raise IndexError(' len x: {0} must be greater than n_past + n_ahead {1}'.format(len(x), n_ahead+n_past))

    if fill is False:
        tuple_list = [tup for _, tup in enumerate(tuple_list) if any_none(tup) is False]
    if all((n_past != 0, n_ahead != 0)):
        x = np.array([v[0] for _, v in enumerate(tuple_list)])
        y = np.array([v[1] for _, v in enumerate(tuple_list)])
        if x.ndim == 1:
            x = np.reshape(x, (1, -1))
        if y.ndim == 1:
            y = np.reshape(y, (1, -1))
        if x.shape[0] != y.shape[0]:
            warn('miss matched rows x: {0} y: {1}'.format(x.shape[0], y.shape[0]))
        if x.shape[1] != n_past:
            warn('n_past miss matched output {0}'.format(x.shape[1]))
        if y.shape[1] != n_ahead:
            warn('n_past miss matched output {0}'.format(n_ahead))
        return x, y
    else:
        x = np.array([v for _, v in enumerate(tuple_list)])
        return x


def n_ahead_gen(x, n_past=0, n_ahead=0, fill_val=None):
    '''
        useage
        x = np.arange(10)
        list(n_ahead_gen(x ,2, 3))
        list(n_ahead_gen(x, 0, 2))
    :param x:list or 1d array
    :param n_ahead: int number of lags ahead
    :param n_past: int number of lags in the past
    :return: past, ahead, index
    '''
    if all((n_past <= 0, n_ahead <= 0)):
        raise IndexError('at least one n_past or n_ahead need to be integers greater than zero')

    x = [fill_val] * n_past + list(x) + [fill_val] * n_ahead
    i = n_past
    while i < len(x) - n_ahead:
        ahead = x[(i):(i+n_ahead)]
        past = x[(i - n_past):i]
        i += 1
        if n_past == 0:
            yield ahead
        elif n_ahead == 0:
            yield past
        else:
            yield past, ahead


class BuildXgboostClassifier:
    ''''
    model_type
    a class that automatically build a classification model using logistic regression
        self.preds_test = model.predict_proba(x_test)[:, 1]
        self.preds_train = model.predict_proba(x_train)[:, 1]
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = bc.feature_names

    '''
    def __init__(self, model_type='nn'):
        '''
        :param model_type: str 'nn', 'ada' build KNeighborsClassifier, AdaBoostClassifier else builds logistic regress
        '''
        import numpy as np
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from scipy.sparse import csc_matrix
        import xgboost as xgb
        np.random.seed(2012)
        bc = load_breast_cancer()
        x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, random_state=2012)
        y_train = np.array(y_train, dtype=np.float)
        y_test = np.array(y_test, dtype=np.float)
        params = {'max_depth': 4,
                  'min_child_weight': 1,
                  'eta': .1,
                  'subsample': .2,
                  'colsample_bytree': .2,
                  'scale_pos_weight': 1,
                  'objective': 'binary:logistic',
                  'eval_metric': "auc"}
        num_boost_round = 20
        x_train = csc_matrix(x_train)
        x_test = csc_matrix(x_test)
        dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=bc.feature_names)
        dtest = xgb.DMatrix(x_test, label=y_test, feature_names=bc.feature_names)
        # build the mode
        model = xgb.train(params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "Train"), (dtest, "Test")],
            early_stopping_rounds=1, verbose_eval=False)

        self.preds_test = model.predict(dtest)
        self.preds_train = model.predict(dtrain)
        self.shaps_train = model.predict(dtrain, pred_contribs=True)
        self.shaps_test = model.predict(dtest, pred_contribs=True)
        self.model = model
        self.dtest = dtest
        self.dtrain = dtrain
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = bc.feature_names


class BinaryClassicationReport:
    ''''
    BinaryClassifcationReport
    a class to simplify documenting model performance,
    returns a txt string that prints well.

    usage:
    n=100
    y_true = np.random.choice([0,1], n)
    y_preds = np.random.normal(size=n)
    id_array = np.random.choice(np.arange(50), n)
    b = BinaryClassicationReport(.5, 'Test', model=np.random.normal, id_name = 'id', feature_names=['f1', 'f2', 'f3'], imp_array=[.1, .2, .4])
    print(b.fit( y_true, y_preds, id_array))

    '''
    def __init__(self, cut_off=0, name=None, file_path=None, model=None, feature_names=None, imp_array=None, id_name=None, n=10):
        '''

        :param cut_off: int or float probablity cut off
        :param name: str name for the text report
        :param file_path: str path to save the report (optional)
        :param model: predictive model (includes type(model) in report (optional)
        :param feature_names: list of feature names (optional)
        :param imp_array: array of feature importance (optional)
        :param id_name: str name for an optional ID Column (when test performances on max prob of an id) (optional)
        :param n int number of top features selected by arg sort (optional)
        '''
        self.params = {'file_path': file_path,
                       'model': model,
                       'name': name,
                       'cut_off': cut_off,
                       'id_name': id_name,
                       'feature_names': feature_names,
                       'imp_array': imp_array,
                       'n': n}
        self.report = None

    def fit(self, y_true, y_preds, id_array=None, comment=None,  **kwargs):
        '''

        :param y_true: array of 0/1 labels
        :param y_preds: array of numeric probablityes
        :param id_array: array of ids
        :param comment: string text to add to report
        :param kwargs:
        :return: string text of classification Report
        '''

        y_true = np.ravel(y_true)
        y_preds = np.ravel(y_preds)
        n_obs = len(y_true)
        if n_obs != len(y_preds):
            raise ValueError(' len(y_true) {0} != len(y_preds) {1} '.format(n_obs, len(y_preds)))

        name = self.params['name']
        model = self.params['model']
        cut_off = self.params['cut_off']
        file_path = self.params['file_path']
        pred_labels = [1 if x > cut_off else 0 for x in y_preds]

        cm = confusion_matrix(y_true, pred_labels)
        cr = classification_report(y_true, pred_labels)

        report = '#################################### \n'
        report = report + '# Final Performance Evaluation "{}"# \n'.format(name)
        report = report + '#################################### \n'
        if isinstance(comment, type(None)) is False:

            report = report + 'comment: {}'.format(comment)

        report = report + '+ ModelType: {} \n'.format(type(model))
        report = report + '+ Run date: {} \n'.format(datetime.datetime.now())
        report = report + '+ Cut off: {} \n'.format(cut_off)
        report = report + '+ N_obs: {} \n'.format(len(pred_labels))
        report = report + '+ {} \n'.format(Counter(y_true))
        report = report + '+ Average prob of label: {} \n'.format(np.mean(y_true))
        report = report + '+ Mean prediction: {} \n'.format(np.mean(y_preds))
        report = report + '+ ROC AUC: {} \n'.format(roc_auc_score(y_true, y_preds))
        report = report + 'Sensitivity (Recall/TPR): {} \n'.format(round(cm[1, 1] / (cm[1, 1] + cm[1, 0]), 3))
        report = report + 'Specificity (selectivity TNR) : {} \n'.format(round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 3))
        report = report + 'Percision positive predictive value (PPV) : {} \n '.format(round(cm[1, 1] / (cm[1, 1] + cm[0, 1]), 3))
        report = report + '\n## Results by Observation  ##\n'
        report = report + '### Confusion Matrix ###\n'
        report = report + str(cm) + '\n'
        report = report + '###  Classifcation Report ###\n'
        report = report + str(cr) + '\n'
        self.report = report

        # adds a sub classication report, where an id_array is specified
        if isinstance(id_array, type(None)) is False:
            self._add_report_by_id(y_true, y_preds, id_array)
        # add featutre importance information
        if isinstance(self.params['feature_names'], type(None)) is False:
            if isinstance(self.params['imp_array'], type(None)) is False:
                self._add_imp()
        # adds information about the models and python env
        self._add_python_info()

        # write to a file
        if isinstance(file_path, type(None)) is False:
            with open(file_path, 'w') as f:
                f.writelines(report)
            print('writting to {}'.format(file_path))

        return self.report

    def _add_report_by_id(self, y_true, y_preds, id_array):
        '''
        Internal method of estimate performance of a model
        on ID col (typically a patient or account ID)
        Aggregates by max prediction
        :param y_true: array of 0/1 labels
        :param y_preds: array of floating proablities
        :param pred_labels: array of predictions 0/1 labels
        :param id_array:  array of ids
        :return: None, sets self.reports
        '''
        n_obs = len(y_true)
        if len(id_array) != n_obs:
            raise ValueError(' len(y_true) {0} != len(id_array) {1} '.format(n_obs, len(id_array)))
        id_name = self.params['id_name']
        self.report = self.report + '## Results, Max Prediction Results by: {} ## \n'.format(id_name)
        cut_off = self.params['cut_off']
        eval_results = pd.DataFrame({'pred': y_preds,
                                     'labels': y_true,
                                     'id': id_array})
        eval_results_grouped = eval_results.groupby(by=['id'])
        y_true_agg = eval_results_grouped['labels'].agg(np.max).astype(np.int)
        y_pred_agg = eval_results_grouped['pred'].agg(np.mean)
        y_pred_label_agg = [1 if x > cut_off else 0 for x in y_pred_agg]
        self.report = self.report + '###  Classification Report by {} ### \n'.format(id_name)
        try:
            self.report = self.report + '+ ROC AUC: {0} by mean prediction per {1} \n'.format(roc_auc_score(y_true_agg, y_pred_agg), id_name)
        except ValueError:
            self.report = self.report + '+ ROC AUC: NULL by mean prediction per {} \n'.format(id_name)

        self.report = self.report + str(classification_report(y_true_agg, y_pred_label_agg))
        self.report = self.report + '### Confusion Matrix ### \n'
        cm = confusion_matrix(y_true_agg, y_pred_label_agg)
        try:
            self.report = self.report + '{} Sensitivity (Recall/TPR): {1} \n'.format(id_name, round(cm[1, 1] / (cm[1, 1] + cm[1, 0]), 3))
            self.report = self.report + '{} Specificity (selectivity TNR) : {1} \n'.format(id_name, round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 3))
            self.report = self.report + '{} Precision positive predictive value (PPV) : {}1 \n '.format(id_name, round(cm[1, 1] / (cm[1, 1] + cm[0, 1]), 3))
        except:
            self.report = self.report + str(cm)

    def _add_python_info(self):
        '''
        Method for adding python env and model information to report
        :return: None
        '''
        import sys
        self.report = self.report + '\n\n #### Module Information'
        self.report = self.report + '\n + python version : {} '.format(sys.version)
        try:
            import FeaturePipe
            self.report = self.report + '\n + featurePipe version : {} '.format(FeaturePipe.__version__)
        except ImportError:
            pass
        try:
            import xgboost
            self.report = self.report + '\n + xgboost: {} '.format(xgboost.__version__)
        except ImportError:
            pass
        try:
            import sklearn
            self.report = self.report + '\n + sklearn: {} '.format(sklearn.__version__)
        except ImportError:
            pass

    def _add_imp(self):
        '''
        Method for adding feature importance information to classification report
        :return: None
        '''
        feature_names = np.array(self.params['feature_names'])
        imp_array = np.array(self.params['imp_array'])
        n_features = len(feature_names)
        n = self.params['n']

        if n_features != len(imp_array):
            raise ValueError(' len features names {0} does not match len imp_array {1}'.format(n_features,
                                                                                               len(imp_array)))
        else:
            args_ordered = np.argsort(imp_array)[::-1]
            n_selected = min([n_features, self.params['n']])
            n_dead_features = np.sum(imp_array == 0)

            index = args_ordered[:n_selected]
            self.report = self.report + '\n\n ## Feature Importance ##'
            self.report = self.report + '\n + num input features: {}'.format(n_features)
            self.report = self.report + '\n + num non contributing features: {}'.format(n_dead_features)
            self.report = self.report + '\n #### Displaying {} Top Features '.format(n_selected)
            for i, j in enumerate(index):
                self.report = self.report + '\n + {0}, {1}: {2} '.format(i, feature_names[j], imp_array[j])


def _gen_sdevs(X, p=.3):
    X = csc_matrix(X)
    for i in range(X.shape[1]):
        array = X[:, i].toarray().flatten()
        if p > np.mean(array == 0):
            output = np.std(array[array != 0])
        else:
            output = 0
        yield output


def sdev_safe(X, p=.2):
    '''
    Caclulates standard deviation across cols of an array
    Has the property that is the col has higher prob of being zero
    than (p), the stdev is assumed to be zero.

    Standard Devition  Safe
    :param X: array or scipy sparce matrix with rows and columns
    :param p: float probablities
    :return: list standard deviations (axis=0) for each column
    '''

    return list(_gen_sdevs(X, p))


def get_importance(X, model, batch_size=10, fill_value=None, sdevs=None, p=.3, alpha=.1):
    '''
    get_importance is a method to estimate feature importance
    Extracts feature importance from sklearn classifiers, or models with .predict() method
    Looks at each col for every predictions, calcuated
        cont =  model.predict(x + sdev * alpha) - model.predict(x - sdev * alpha)

    param: X numpy array or scipy sparse matrix of features
    param: model, sklearn classifier, xgboost model or booster or any model that uses a .predict() method.
    param: batch_size int size of the batches to run through the predict method
    param: fill_value int, float, list or array of values to fill.  if an iterable is used, it must have the same
        len as X.shape[1]
    param: p float probability, that a feature is zero (if is zero more, then treated as one hot)
    param: alpha float (multiplied by standard dev of the field
    returns: scipy sparse matrix of shape X.shape, where the values are feature contributions
    notes:
        values = prediction_original_data -   prediction_with_one_column_replaced

    usage:

        bc = load_breast_cancer()
        x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, random_state=2012)
        model = xgb.XGBClassifier()
        model =  model.fit(x_train, y_train)
        get_importance(X_train, model)

    '''

    # helper functions
    # yields one row at a time from a scipy sparse arrayt
    def _row_gen(X):
        n_rows = X.shape[0]
        for i in range(n_rows):
            yield X[i]

    # yields one batch at a time (subset of rows) from a sparse array
    def _batch_gen(X, batch_size=10, n_batches=None):
        n_rows = X.shape[0]
        n_batches = min(n_rows, int(n_rows / batch_size) + 1)
        g = _row_gen(X)
        for i in range(n_batches):
            output = []
            for _ in range(batch_size):
                try:
                    output.append(next(g))
                except StopIteration:
                    pass
            try:
                yield vstack(output)
            except ValueError:
                break

    # yields one array of feature importance (replacing values with fill value)
    def _imp_gen(X, model, batch_size, sdevs):
        g = _batch_gen(X, batch_size)
        n_rows = X.shape[0]
        n_cols = X.shape[1]
        assert len(sdevs) == n_cols
        n_batches = min(n_rows, int(n_rows / batch_size) + 1)
        for _ in range(n_batches):
            x = next(g).toarray()
            output = np.zeros(x.shape)
            for i in range(n_cols):
                old_x = np.copy(x)
                new_x = np.copy(x)
                val = sdevs[i]
                if val != 0:
                    new_x[:, i] = x[:, i] - (val * alpha)
                    old_x[:, i] = x[:, i] + (val * alpha)
                else:
                    old_x[:, i] = 0
                try:
                    v_old = model.predict_proba(old_x)[:, 1]
                    v_new = model.predict_proba(new_x)[:, 1]
                except AttributeError:
                    try:
                        v_old = model.predict(old_x)
                        v_new = model.predict(new_x)
                    except AttributeError:
                        v_old = model.predict(xgb.DMatrix(old_x), validate_features=False)
                        v_new = model.predict(xgb.DMatrix(new_x), validate_features=False)
                influence = v_new - v_old
                output[:, i] = influence
            yield csc_matrix(output)

    # actually method running (using helper methods )
    n_rows = X.shape[0]
    X_sparse = csc_matrix(X)
    if isinstance(sdevs, type(None)):
        sdevs = list(_gen_sdevs(X, p))
    g = _imp_gen(X_sparse, model, batch_size, sdevs)
    return vstack(list(g))

# test methods

def _get_importance_test():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    bc = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, random_state=2012)
    y_train = np.array(y_train, dtype=np.float)
    params = {'max_depth': 4,
              'scale_pos_weight': 1,
              'eval_metric': "auc",
             'colsample_bytree':.4}
    X_train = csc_matrix(x_train)
    X_test = csc_matrix(x_test)

    # build the mode
    model = xgb.XGBClassifier(**params)
    model =  model.fit(X_train, y_train)

    imp = get_importance(X_test, model ,batch_size=10, alpha=.8)
    logger.info('get_importance test completed')

def _binary_classifcation_report_test():
    y_true = [ 0, 0, 1, 0, 1, 1]
    y_preds = [.3, .1, .3, .2, .5, .1]
    id = [0, 0, 0, 1, 1, 1]
    r = BinaryClassicationReport(cut_off=.2, file_path=None)
    r.fit(y_true, y_preds, id_array=id)
    logger.info('binary classification test completed')
