if __name__ == "__main__":
    import logging
    from ClarityUtils import setup_logger
    from ClarityUtils.utils import _utils_test
    logger = logging.getLogger('ClarityUtils.test')
    logger.setLevel(logging.DEBUG)
    _utils_test()
    logger.info('ClarityUitls TestCompleted')