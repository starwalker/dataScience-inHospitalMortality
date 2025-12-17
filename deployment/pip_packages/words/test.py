
from words.preprocessing import load_stop_words, load_abbreviations
import logging
from FeaturePipe import setup_logger
logger = logging.getLogger('words.test')


def Cleaner_test():
    from datetime import datetime
    from sklearn.datasets import fetch_20newsgroups
    from words import Cleaner
    st = datetime.now()
    c = Cleaner(stop_words=True, stem=True)
    data = ['This is my tests data ', 'this more testing data', None, 1, '02']
    results = c.fit_transform(data)
    t1 = results[1] == results[0]
    results = c.fit_transform(data, params={'stop_words': True, 'stem': True})
    t2 = results[1] == results[0]
    t3 = c.fit_transform(['this is a strings tb'], replace_abbreviations=True, rm_numbers=False, rm_stop_words=True,
                         stem=True) == ['string tuberculosi']
    if all((t1, t2, t3)) is False:
        raise ValueError('validation test failed')
    logger.info('completed in:' + str(datetime.now() - st))
    from datetime import datetime
    text = fetch_20newsgroups().data
    st = datetime.now()
    c = Cleaner(stop_words=True, stem=True)
    results = c.fit_transform(text[1:100])
    logger.info(results[0:2])
    logger.info('completed in:' + str(datetime.now() - st))
    logger.info('words test complete')

logging.info('loading abreviations .. ')
abv = load_abbreviations()
logger.info('abv: ' + str(abv))
logger.info('loading top words ... ')
stops = load_stop_words()
logger.info('stops :' + str(stops))
logger.info('runing cleaner test ...')
Cleaner_test()

