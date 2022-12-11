import json
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    args = {
        'output_dir': './pretrained/swig/v3',
        'model_type': 'mgsrtr',
    }
    output_dir = Path(args['output_dir'])

    summary_dir = output_dir / 'summary' / str(args['model_type'])
    writer = SummaryWriter(str(summary_dir))

    train_stats = {
        14: {
            'loss': 19.343439108623407,
        },
        15: {
            'loss': 19.30047102638822,
        },
        16: {
            'loss': 19.24439591941228,
        },
        17: {
            'loss': 19.19564075603865,
        },
        18: {
            'loss': 19.13595733104226,
        },
        19: {
            'loss': 19.06203988842677,
        },
        20: {
            'loss': 18.993439108623407,
        },
    }

    test_stats = {
        14: {
            'loss': 22.00855380822742,
        },
        15: {
            'loss': 22.03110803001661,
        },
        16: {
            'loss': 21.97741906510527,
        },
        17: {
            'loss': 21.90929350415974,
        },
        18: {
            'loss': 21.95581940354333,
        },
        19: {
            'loss': 21.90231533203868,
        },
        20: {
            'loss': 21.94849263186678,
        },
    }

    print(writer.log_dir)
    for epoch in range(14, 21):
        print('training: {}, validation: {}'.format(train_stats[epoch]['loss'], test_stats[epoch]['loss'] ))
        writer.add_scalars('epoch_loss', {
            "training": train_stats[epoch]['loss'],
            "validation": test_stats[epoch]['loss'],
        }, epoch)

    writer.close()