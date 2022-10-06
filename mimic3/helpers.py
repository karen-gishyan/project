import logging

def configure_logger():
    logger = logging.getLogger('mimic3')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('../info.log',mode='w')
    # file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger