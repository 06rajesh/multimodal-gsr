import json

from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import pandas as pd

def sort_dict(x):
    return dict(sorted(x.items(), key=lambda item: item[1], reverse=True))

def display_graphs_from_json(json_file):
    with open(json_file) as f:
        all_stats = json.load(f)

    verbs_count = sort_dict(all_stats['verbs'])
    nouns_count = sort_dict(all_stats['nouns'])
    roles_count = sort_dict(all_stats['roles'])
    print(nouns_count)

if __name__ == '__main__':
    logdir = Path('./flicker30k/pretrained/v7/')
    log_file = logdir / 'log_stats.txt'
    display_graphs_from_json(log_file)