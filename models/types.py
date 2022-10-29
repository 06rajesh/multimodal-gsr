from enum import Enum

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ModelType(Enum):
    MGSRTR:str = 'mgsrtr'
    DuelEncGSR:str = 'duel_enc_gsr'
    T5_MGSRTR:str = 't5_mgsrtr'

    def __str__(self):
        return str(self.value)

    @staticmethod
    def from_str(type_str:str):
        if type_str.lower() == 'mgsrtr':
            return ModelType.MGSRTR
        elif type_str.lower() == 'duel_enc_gsr':
            return ModelType.DuelEncGSR
        elif type_str.lower() == 't5_mgsrtr':
            return ModelType.T5_MGSRTR
        else:
            raise NotImplementedError