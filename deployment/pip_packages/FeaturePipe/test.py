import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
import logging
import pickle
import os
from FeaturePipe.utils import get_lagged_array, _binary_classifcation_report_test, _get_importance_test
from FeaturePipe.pipeline import FeatureName, ColumnSelect
from FeaturePipe.feature_extraction import TextFeature, Text2List
from FeaturePipe.pipeline import Bagpipe, BagpipeDeploy, col_sub_setter
from FeaturePipe.preprocessing import bp, temp, weight, string_to_float, _medicalCleanerTest
from FeaturePipe import setup_logger
from FeaturePipe.locations import _locations_test
import warnings
if __name__ == "__main__":
    logger = logging.getLogger('FeaturePipe')
    logger.setLevel(logging.INFO)


def TextFeature_test():
    text = fetch_20newsgroups().data
    data = pd.DataFrame({'x': text[1:] + [None], 'y': text[:-1] + [1]})
    n = 50
    t = TextFeature(back_end='sklearn', max_features=n)
    t.fit(data.head(100), verbose=True)
    features = t.transform(data.head())
    if features.shape[1] == n:
        logger.info('text feature test 1 passed')
    else:
        logger.error('test feature 1 test failed')
        raise RuntimeError()


def TexFeature_test_2():
    np.random.seed(2012)
    n = 75
    num_words = 3
    text = np.array(['a :: b :: c', 'a', 'b', 'c :: b', 'a :: c', 'd', None, 1])
    data = np.random.choice(text, n, replace=True)
    t = TextFeature(back_end='keras', clean=False, split='::', num_words= num_words)
    features = t.fit_transform(data)
    if all((t.get_feature_names() == ['a', 'c', 'b'], features.shape == (n,  num_words))):
        logger.info('test feature test 2 passed')
    else:
        logger.error('test feature test 2 failed')
        raise RuntimeError()


def TextFeature_test_3():
    logger.debug('running text feature test of news group 20')
    text = fetch_20newsgroups().data
    data = pd.DataFrame({'x': text[1:] + [None], 'y': text[:-1] + [1]})
    n = 50
    t = TextFeature(back_end='sklearn', max_features=n)
    t.fit(data.head(100), verbose=True)
    features = t.transform(data.head())
    logger.debug('text feature test on news group 20 competed')
    logger.debug('running text feature test on generated data')
    np.random.seed(2012)
    n = 75
    num_words = 3
    text = np.array(['a :: b :: c', 'a', 'b', 'c :: b', 'a :: c', 'd', None, 1])
    data = np.random.choice(text, n, replace=True)
    t = TextFeature(back_end='keras', clean=False, split='::', num_words= num_words)
    features = t.fit_transform(data)
    if all((t.get_feature_names() == ['a', 'c', 'b'], features.shape == (n,  num_words))):
        logger.info('text feature test completed')
    else:
        logger.error('text feature test completed, howeverget_feauture_names failed')
        raise RuntimeError()


def FeatureName_test():
    '''
    Test test  whether or FeatureNames class inside pipes is working
    :return: logical
    '''
    f = FeatureName()
    text = fetch_20newsgroups().data[1:500]
    n = len(text)
    np.random.seed(2012)
    data = pd.DataFrame({'text_col': text,
                         'num_col1': np.arange(n),
                         'num_col2': np.arange(n),
                         'cat_col1': np.random.choice(['a', 'bb', 'c', 'd'], n, replace=True),
                         'cat_col2': np.random.choice(['a', 'b', 'c', 'd'], n, replace=True)})
    text_col = 'text_col'
    num_col = ['num_col1', 'num_col2']
    cat_col = ['cat_col1', 'cat_col2']
    text_pipe = Pipeline([('select', ColumnSelect(text_col)), ('textlist', Text2List()),
                          ('text', CountVectorizer(min_df=.01, max_df=.7, max_features=4))])

    num_pipe = Pipeline([('select', ColumnSelect(num_col)), ('numeric', SimpleImputer(strategy='median')),
                         ('pca', PCA(1))])
    cat_pipe = Pipeline([('select', ColumnSelect(cat_col)), ('numeric', OneHotEncoder())])
    cat_pipe.fit(data)
    num_pipe.fit(data)
    f = FeatureName()
    feature_names = list(f.get_feature_names(cat_pipe))
    if feature_names == ['cat_col1_a','cat_col1_bb', 'cat_col1_c', 'cat_col1_d', 'cat_col2_a', 'cat_col2_b',
                         'cat_col2_c',
                         'cat_col2_d']:
        logger.info('feature names test on pipeline cat encoder passed')
    else:
        logger.error('feature name test failed on cat encoder pipe')
        raise RuntimeError()

    input_names = f.get_input_names(cat_pipe)
    feature_names = list(f.get_feature_names(num_pipe))
    if feature_names == ['pca0']:
        logger.info('feature names test on num pipe passed')
    else:
        logger.error('feature name test failed on num pipe pca ')
        raise RuntimeError()


    pipe_list = [('text_pipe', text_pipe), ('cat_pipe', cat_pipe), ('num_pipe', num_pipe)]
    pipe = FeatureUnion(transformer_list=pipe_list)
    pipe.fit(data)
    f = FeatureName()
    x = list(f.get_feature_names(pipe))
    missing = set(['text_col.edu', 'text_col.for', 'text_col.on', 'text_col.you', 'cat_col1_a', 'cat_col1_bb',
                   'cat_col1_c', 'cat_col1_d', 'cat_col2_a', 'cat_col2_b', 'cat_col2_c', 'cat_col2_d', 'pca0']).difference(set(x))
    if not missing:
        logger.info(' feature names test passed')
    else:
        logger.error('feature names did not output the expected feature name list')
        logger.debug(' feature name output, missing: ' + str(missing))
        raise RuntimeError()
    y = f.get_input_names(pipe)

    missing = set(y).difference(set(['text_col', 'cat_col1', 'num_col1', 'num_col2', 'cat_col2']))
    if not missing:
        logger.info(' feature names class  input names method test passed')
    else:
        logger.error('feature names. did not output the expected feature name list')
        logger.debug(' feature name input names method output ' + str(missing))


def lagged_array_test():
    x = np.arange(5)
    z = list(get_lagged_array(x, 2, 1))
    get_lagged_array(x, 1, 2, fill=True)
    get_lagged_array(x, 2, 1, fill=True)
    logger.info('lagged array run completed')


def decilizer_test():
    '''
    self contained test for the decilizer
    tests the mean and the order.
    call: decilizer_test()
    :return:
    '''
    from FeaturePipe.feature_extraction import Decilizer
    import numpy as np
    np.random.seed(2012)
    n = 100
    x = np.arange(n) + np.random.normal(0, 1, n)/100
    y = x[::-1]
    data = np.column_stack((x, y))
    d = Decilizer()
    d.fit(data)
    output = d.transform(data)
    if 50.0 == np.round(np.mean(output)):
        logger.debug('deciler mean test passed')
    else:
        logger.error('deciline mean test failed, mean: ' + str(np.mean((output))))
        raise RuntimeError()
    if all(output[:, 0] == output[::-1, 1]):
        logger.debug('decilier revers test passed')
    else:
        logger.error('decile reverser test failed deciles for 1,2,3 != [3,2,1][::-1]')
        raise RuntimeError()
    if data.shape == output.shape:
        logger.debug('deciler shape test passed')
    else:
        logger.error('deciler shape test failed, in: ' + str(data.shape) + ' out: ' + str(output.shape))
        raise RuntimeError()
    d.fit(x)
    z = d.transform(x)
    d.fit([1,2,3])
    z = d.transform([1,2,3])
    logger.info('deciler test completed')


def ColumnSelect_test():
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['this', 'is', 'a test']})
    logger.info('col select data frame test')
    c = ColumnSelect(col_list=['col1'])
    c.fit(data)
    c.transform(data)
    c.transform(data.iloc[1, :])

    c = ColumnSelect(col_list='col1')
    c.fit(data)
    f = c.transform(data)
    feature_names = c.get_feature_names()
    c = ColumnSelect()
    c.fit(data.values)
    c.transform(data.values)
    c.transform(data.values[1, :])
    logger.info('Col select test completed')


def bagpipe_test():
    b = load_breast_cancer()
    X = b.data
    y = np.array(b.target, dtype=np.float)
    feature_names = list(b.feature_names)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    pipe.fit(X)
    features = pipe.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=.33, random_state=2012)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    # fit model no training data
    params = {'max_depth': 2, 'eta': .3, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    model = xgb.train(params, dtrain, evals=[(dtest, 'test')], verbose_eval=False)
    preds = model.predict(dtrain)

    logger.debug('testing bagpipe init')
    bp = Bagpipe(model=model, pipe=pipe, data=X, preds=preds)
    logger.debug('testing bagpipe predict')
    p = bp.predict(X[0:10, :])
    logger.debug('testing context reducer ')
    g = bp._contrib_reducer_gen()
    feature_names_1, features_1, contributions_1, preds_1 = next(g)
    # test summary gen
    logger.debug('testing context summary gen')
    g = bp._summary_gen(feature_names_1, features_1, contributions_1, preds_1)
    summary = list(g)

    logger.debug('testing bagpipe standard output gen')
    g = bp._standard_output_gen()
    x = next(g)
    # test standard predict


    logger.debug('testing bagpipe transform and prediction')
    features = bp.transform(X[18:20, :])
    z = bp.predict_standard_output(features=features)

    logger.debug('testing bagpipe one row prediction')
    features = bp.transform(X[19:20, :])
    z = bp.predict_standard_output(features=features)

    logger.debug('testing serialization')
    file_name = 'test_file.p'
    bp.save(file_name)
    with open(file_name, 'rb') as f:
        bp_loaded = pickle.load(f)
    features = bp_loaded.transform(X[18:20, :])
    x = bp_loaded.predict_standard_output(features=features)
    bp.get_pipe_key()
    logger.debug('serialization test passed')
    logger.info('bagpipe test completed')
    try:
        os.system('rm ' + file_name)
    except:
        pass


def bagpipe_test_pd():
    # runs test on a pandas data frame
    b = load_breast_cancer()
    feature_names = list(b.feature_names)
    X = pd.DataFrame(b.data, columns=feature_names)
    y = np.array(b.target, dtype=np.float)

    pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    pipe.fit(X)
    features = pipe.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=.33, random_state=2012)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    # fit model no training data
    params = {'max_depth': 2, 'eta': .3, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    model = xgb.train(params, dtrain, evals=[(dtest, 'test')], verbose_eval=False)
    preds = model.predict(dtrain)

    logger.debug('testing bagpipe init')
    bp = Bagpipe(model=model, pipe=pipe, data=X, preds=preds)
    logger.debug('testing bagpipe predict')
    p = bp.predict(X.iloc[0:10, :])
    logger.debug('testing context reducer ')
    g = bp._contrib_reducer_gen()
    feature_names_1, features_1, contributions_1, preds_1 = next(g)
    # test summary gen
    logger.debug('testing context summary gen')
    g = bp._summary_gen(feature_names_1, features_1, contributions_1, preds_1)
    summary = list(g)

    logger.debug('testing bagpipe standard output gen')
    g = bp._standard_output_gen()
    x = next(g)
    # test standard predict
    o = bp.predict_standard_output(data=X.iloc[1:5, :], cols=feature_names[0:1])
    if feature_names[0] in o.columns:
        logger.debug('cols check for stardard output predict passed')
    else:
        logger.error('cols are not being passed through stardard output predict')
        raise RuntimeError()
    logger.debug('testing bagpipe transform and prediction')
    o = bp.predict_standard_output(data=X.iloc[1:5, :], cols=feature_names[1:3])
    missing = set(feature_names[1:3]).difference(set(o.columns))
    if missing:
        logger.error('cols are not being passed through stardard output predict')
        raise RuntimeError()
    else:
        logger.debug('cols successfully passed in predict standard output predict')
    logger.debug('testing bagpipe transform and prediction')
    features = bp.transform(X.iloc[18:20, :])
    z = bp.predict_standard_output(features=features)

    logger.debug('testing bagpipe one row prediction')
    features = bp.transform(X.iloc[19:20, :])
    z = bp.predict_standard_output(features=features)

    bp.get_pipe_key()
    logger.info('bagpipe test completed')


def col_sub_setter_test():
    data = pd.DataFrame({'y': [1, 2, 3], 'x': [7, 8,  9]})
    col_sub_setter(data, 'x')
    col_sub_setter(data, ['x', 'y'])
    data = np.array([[1, 2], [3,  4], [5, 6]])
    col_sub_setter(data, [1])
    col_sub_setter(data, [1, 1, 1, 0]) == np.array([[2, 2, 2, 1], [4, 4, 4, 3], [6, 6, 6, 5]])


def bagpipe_deploy_test():
    b = load_breast_cancer()
    X = b.data
    y = np.array(b.target, dtype=np.float)
    feature_names = list(b.feature_names)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    pipe.fit(X)
    features = pipe.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=.33, random_state=2012)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    # fit model no training data
    params = {'max_depth': 2, 'eta': .3, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    model = xgb.train(params, dtrain, evals=[(dtest, 'test')], verbose_eval=False)
    preds = model.predict(dtrain)
    bp0 = Bagpipe(model=model, pipe=pipe, data=X, preds=preds)
    bp1 = Bagpipe(model=model, pipe=pipe, data=X, preds=preds)

    pipe = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    pipe.fit(X)
    bp2 = Bagpipe(model=model, pipe=pipe, data=X, preds=preds)
    d = BagpipeDeploy(bagpipes=[bp0, bp1, bp2])
    if d.input_cols:
        logger.debug(' bp deploy setup input cols' + str(d.input_cols))
    else:
        logger.error('bp deploy input col setup failed ')
    logger.debug('testing bpd predict gen')
    g = d._predict_gen(X)
    z = list(g)
    logger.debug('testing bpd predict gen completed')
    logger.debug('testing bpd predict method')
    outputs = d.predict(X)
    logger.debug('bpd predict method with input shape: {0},  output shape: {1}'.format(X.shape, outputs.shape))
    if outputs.shape[0] != X.shape[0] * len(d.bagpipes):
        logger.error('miss matched bpd output shape and input shape (should be n_models * input rows')
    else:
        logger.info('bp deploy predict method row count passed')


def text_2_list_test():
    t = Text2List(join_char=' : ' )
    x = ['this is text', 'and more text']
    y = [12, None]
    t.fit(x)
    if len(t.transform(x)) == len(x):
        logger.info('test to list passed list test')
    else:
        logger.error('test to list list method error')
    d = pd.DataFrame({'y': y, 'x': x})
    t = Text2List(join_char=':')
    t.fit(d)
    z = t.transform(d)
    if z == ['12.0:this is text', 'nan:and more text']:
        logger.debug('test to list data frame check passed')
    else:
        logger.error('test to list pd method error')
    t = Text2List(join_char=':')
    t.fit(pd.Series(x))
    z = t.transform(x)
    if z == ['this is text', 'and more text']:
        logger.debug('text to list series test passed')
    else:
        logger.error('test to list pd Series method error')
    logger.info('text to list tests completed')

    def string_to_float_test():
        if all((string_to_float('123') == 123,
                string_to_float('8') == 8,
                string_to_float('100.5') == 100.5,
                string_to_float('lasdfg123 mlm') == 123,
                string_to_float('lasdfg123 mlm') == 123,
                string_to_float('lasdfg123.12mlm') == 123.12)):
            logger.info('float cleaner test passed ')
        else:
            logger.error('float cleaner test failed ')
            raise ValueError('float cleaner test failed')


def weight_test():
    if all((weight('10kg', kg=False, na_val=0) == 22.04623,
            weight('100lbs', kg=True, na_val=0) == 220.46230000000003,
            weight('sdf83.0lbs', kg=False, na_val=0) == 83.0,
            weight('83.0 lbs 1.2 oz', kg=False, na_val=0) == 83.075,
            weight('83lbs 8oz', kg=False, na_val=0) == 83.5)):
        logger.info('weight cleaner test passed ')
    else:
        logger.error('weight cleaner test failed ')
        raise ValueError('weight cleaner test failed')


def bp_test():
    if all((bp('110/90') == 110,
            bp('110/90', systolic=False) == 90,
            bp('asd110/90', systolic=False) == 90)):
        logger.info('bp cleaner test passed')
    else:
        logger.error('bp cleaner test failed ')
        raise ValueError('bp cleaner test failed ')


def temp_test():
    if all((temp("36.8 °C (98.2 °F)", c=True) == 36.8,
            temp("36.8 °C (98.2 °F)", c=False) == 98.24,
            temp("sdf36.8 °C", c=False) == 98.24)):

        logger.info('tempurature cleaner test passed')
    else:
        logger.error('tempurature cleaner test failed ')
        raise ValueError('tempurature cleaner test failed ')


def string_to_float_test():
    if all((string_to_float('123') == 123,
            string_to_float('8') == 8,
            string_to_float('100.5') == 100.5,
            string_to_float('lasdfg123 mlm') == 123,
            string_to_float('lasdfg123 mlm') == 123,
            string_to_float('lasdfg123.12mlm') == 123.12)):
        logger.info('float cleaner test passed ')
    else:
        logger.error('float cleaner test failed ')
        raise ValueError('float cleaner test failed')


if __name__ == "__main__":
    _locations_test()
    TextFeature_test()
    text_2_list_test()
    TextFeature_test_3()
    ColumnSelect_test()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        bagpipe_test()
        bagpipe_deploy_test()
    FeatureName_test()
    decilizer_test()
    string_to_float_test()
    weight_test()
    bp_test()
    temp_test()
    _medicalCleanerTest()
    _binary_classifcation_report_test()
    _get_importance_test()
    logger.info('feature pipe test battery complete')
