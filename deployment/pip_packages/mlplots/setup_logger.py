import logging
name = 'mlplots'
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(name)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
