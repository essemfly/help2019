class LocalConfig:
    env = 'localhost'
    TRAIN_DIR = 'sample'
    VOLUME_DIR = 'src/volume'
    LOG_DIR = 'src/logs'
    TEST_DIR = 'sample'
    OUTPUT_DIR = 'src/volume'


class ProdConfig:
    env = 'localhost'
    TRAIN_DIR = '/data/train'
    VOLUME_DIR = '/data/volume'
    LOG_DIR = '/data/volume/logs'
    TEST_DIR = '/data/test'
    OUTPUT_DIR = '/data/output'
