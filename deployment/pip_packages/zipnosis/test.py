if __name__ == "__main__":
    from zipnosis.utils import  _test_utils
    from zipnosis.transformers  import _test_zipnosis_transformer
    from zipnosis.setup_logger import logger
    import logging
    logger.setLevel(logging.DEBUG)
    _test_utils()

    _test_zipnosis_transformer()
