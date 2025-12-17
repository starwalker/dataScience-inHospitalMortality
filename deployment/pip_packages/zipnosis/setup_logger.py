import logging
# base logger setup, to standardize logging across classes
name = 'zipnosis'
formatter = logging.Formatter(fmt='%(asctime)s -  %(name)s - %(levelname)s  - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(name)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)
