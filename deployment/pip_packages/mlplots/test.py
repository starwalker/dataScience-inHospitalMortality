from mlplots.plot_methods import ShaplyPlot
from FeaturePipe.utils import BuildXgboostClassifier
from mlplots.plot_methods import ClassificationPlot
from FeaturePipe.utils import class_performance, optimum_cutoff
import numpy as np
import random
import logging
from FeaturePipe import FeatureName, ColumnSelect, Text2List
from mlplots import setup_logger
logger = logging.getLogger('mlplots.test')
logger.setLevel(logging.DEBUG)


def Shaply_plot_test():
    b = BuildXgboostClassifier()
    shaps = b.shaps_test
    features = b.x_test
    feature_names = b.feature_names
    s = ShaplyPlot(shaps, features, feature_names)
    s.plot('worst concave points')
    #s.plot('worst concave points',ylim=(-1,1))
    logger.info('shapely plot test completed')

def Shapely_Group_test():
    import pandas as pd
    import xgboost as xgb
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.decomposition import PCA
    f = FeatureName()
    text = fetch_20newsgroups().data[1:500]
    n = len(text)
    np.random.seed(2012)
    target = fetch_20newsgroups(categories=['alt.atheism', 'sci.med']).target[1:500]
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
    pipe_list = [('text_pipe', text_pipe), ('cat_pipe', cat_pipe), ('num_pipe', num_pipe)]
    pipe = FeatureUnion(transformer_list=pipe_list)
    pipe.fit(data)
    features = pipe.transform(data)
    f = FeatureName()
    feature_names = list(f.get_feature_names(pipe))
    y = f.get_input_names(pipe)
    params = {'max_depth': 3,
              'min_child_weight': 10,
              'eta': .3,
              'subsample': 1,
              'colsample_bytree': .8,
              'scale_pos_weight': 1,
              'objective': 'binary:logistic',
              'eval_metric': "auc"}

    dtrain = xgb.DMatrix(features, label=target, feature_names=feature_names)
    model = xgb.train(params, dtrain, num_boost_round=2)
    shaps = model.predict(dtrain, pred_contribs=True)

    print(shaps.shape)
    print(features.shape)
    print(feature_names)
    s = ShaplyPlot(shaps, features, feature_names)
    s.plot_columns('text_col')
    #s.plot('cat_col1_a', jitter=0.02)

def ClassifcationPlot_test():
    b = BuildXgboostClassifier()
    train = np.array((b.y_train, b.preds_train))
    test = np.array((b.y_test, b.preds_test))
    #c = ClassificationPlot(b.y_train, b.preds_train, b.y_test, b.preds_test)
    c = ClassificationPlot(train, test)
    class_performance(b.y_train, b.preds_train)
    optimum_cutoff(b.y_train, b.preds_train, key='mcc')
    c.plot_sense_spec()
    c.plot_auc()
    c.plot_performance_bars()
    c.plot_cutoff_metric()
    c.plot_confusion_matrix()

    # ---------------generate fake group list size of features for testing purposes ------------
    departments = ['ICU', 'PEDIATRIC', 'NEUROLOGY', 'EMERGENCY', 'ONCOLOGY', 'SURGICAL TRAUMA', 'CARDIOVASCULAR']
    groups = []
    labels = train[0]
    for feature in labels:
        index = random.randint(0, 6)
        groups.append(departments[index])
    # --------------------------------------------------------------------------------------------
    key = 'auc'
    c.plot_groups(groups, key)
    logger.info('classification plot test complete')


def test_interaction_plot():
    from FeaturePipe.utils import BuildXgboostClassifier
    from mlplots.plot_methods import Interaction
    b = BuildXgboostClassifier()
    i = Interaction(features=b.x_test.toarray(), feature_names=b.feature_names, xgb_model=b.model, steps=10 )
    i.plot('worst concave points', 'worst area')



if __name__ == "__main__":
    ClassifcationPlot_test()
    Shaply_plot_test()
    Shapely_Group_test()
    test_interaction_plot()
    test_interaction_plot()