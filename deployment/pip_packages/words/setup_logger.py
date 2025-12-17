import logging
name = 'words'
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(name)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)
