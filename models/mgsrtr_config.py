import os
from dotenv import load_dotenv
import json

from pathlib import Path

from .types import ModelType

class MGSRTRConfig:
    lr:float
    lr_backbone: float
    lr_drop: float
    weight_decay: float
    clip_max_norm: float
    distributed: bool
    batch_size: int
    backbone: str
    position_embedding: str
    max_sentence_len: int
    enc_layers: int
    dec_layers: int
    dim_feedforward: int
    hidden_dim: int
    dropout: float
    nheads: int
    noun_loss_coef: int
    verb_loss_coef: int
    bbox_loss_coef: int
    bbox_conf_loss_coef: int
    giou_loss_coef: int
    dataset_file: str
    swig_path: str
    flicker_path: str
    dev: bool
    test: bool
    inference: bool
    output_dir: str
    device: str
    seed: int
    epochs: int
    start_epoch: int
    resume: bool
    num_workers: int
    saved_model: str
    world_size: int
    dist_url: str
    model_type: ModelType

    def __init__(self,
                 dataset:str='flicker30k',
                 swig_path:str ='./flicker30k',
                 flicker_path:str  = './SWiG',
                 device:str ='cpu',
                 resume:bool = False,
                 start_epoch:int = 0,
                 num_workers:int = 4,
                 version:str = 'V1',
                 model_type_str:str = ModelType.MGSRTR.value,
            ):

        if dataset.lower() == 'flicker30k':
            dataset_path = flicker_path
        elif dataset.lower() == 'swig':
            dataset_path = swig_path
        else:
            raise ValueError("Invalid dataset type. Only supports `flicker30k` and `swig` dataset type")

        root = Path(dataset_path)
        output_dir = root / 'pretrained' / version
        saved_model_path = output_dir / 'checkpoint.pth'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_type = ModelType.from_str(model_type_str)

        self.lr = 0.0001
        self.lr_backbone = 1e-5
        self.lr_drop = 100
        self.weight_decay = 0.0005
        self.clip_max_norm = 0.1
        self.distributed = False
        self.batch_size = 16
        self.backbone = 'resnet50'
        self.position_embedding = 'learned'
        self.max_sentence_len = 100
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.hidden_dim = 512
        self.dropout = 0.25
        self.nheads = 8
        self.noun_loss_coef = 1
        self.verb_loss_coef = 1
        self.bbox_loss_coef = 5
        self.bbox_conf_loss_coef = 5
        self.giou_loss_coef = 5
        # dataset_file='swig',
        self.dataset_file = dataset
        self.swig_path = swig_path
        self.flicker_path = flicker_path
        self.dev = False
        self.test = False
        self.inference = False
        self.output_dir = str(output_dir)
        self.device = device
        self.seed = 42
        self.epochs = 40
        self.start_epoch = start_epoch
        self.resume = resume
        self.num_workers = num_workers
        self.saved_model = str(saved_model_path)
        self.world_size = 1
        self.dist_url = 'env://'
        self.model_type = model_type

    @classmethod
    def from_env(cls):
        load_dotenv()

        dataset = os.getenv('DATASET', 'flicker30k')
        device = os.getenv('DEVICE', 'cpu')
        dataset_path = os.getenv('DATASET_PATH', './flicker30k')
        resume_str = os.getenv('RESUME', "False")
        start_epoch = int(os.getenv("START_EPOCH", "0"))
        num_workers = int(os.getenv("NUM_WORKERS", "4"))
        version = os.getenv("VERSION", "V1")
        model_type_str = os.getenv("MODEL_TYPE", "mgsrtr")

        resume = False
        if resume_str.lower() == "true":
            resume = True

        flicker_path = './flicker30k'
        swig_path = './SWiG'

        if dataset == 'flicker30k':
            flicker_path = dataset_path
        elif dataset == 'swig':
            swig_path = dataset_path

        return cls(
            dataset=dataset,
            device=device,
            flicker_path=flicker_path,
            swig_path=swig_path,
            start_epoch=start_epoch,
            num_workers=num_workers,
            version=version,
            resume=resume,
            model_type_str=model_type_str
        )

    @classmethod
    def from_config(cls, config_path:str):

        default_conf = cls.from_env()
        env_keys = ['dataset', 'device', 'flicker_path', 'swig_path', 'start_epoch', 'num_workers', 'version', 'resume', 'model_type_str']

        # Opening JSON file
        f = open(config_path)

        configs = json.load(f)

        f.close()

        for item in configs:
            if item not in env_keys and hasattr(default_conf, item):
                val = getattr(default_conf, item)
                config_val = configs[item]
                if isinstance(val, bool):
                    config_val = True if configs[item].capitalize() == 'True' else False
                elif isinstance(val, int):
                    config_val = int(configs[item])
                elif isinstance(val, float):
                    config_val=  float(configs[item])

                setattr(default_conf, item, config_val)

        setattr(default_conf, 'model_type', ModelType.from_str(configs['model_type']))

        return default_conf

    def save_config(self, path:str=None):

        config = {}
        exclude = ['SWiG_json_train', 'idx_to_verb', 'idx_to_role', 'idx_to_class', 'vidx_ridx', 'SWiG_json_dev']

        for attr, value in self.__dict__.items():
            if attr not in exclude:
                config[attr] = str(value)

        config['num_verbs'] = str(len(self.__dict__['idx_to_verb']))
        config['num_roles'] = str(len(self.__dict__['idx_to_role']))

        if not path:
            path = self.output_dir

        path = Path(path)
        target_file = path / 'config.json'

        with open(target_file, 'w') as f:
            json.dump(config, f, indent=2)