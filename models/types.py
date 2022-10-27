from enum import Enum

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ModelType(Enum):
    MGSRTR:str = 'mgsrtr'
    DuelEncGSR:str = 'duel_enc_gsr'
    T5MGSRTR:str = 't5_mgsrtr'
