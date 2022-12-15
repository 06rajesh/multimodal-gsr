import json
import re
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn

# function to create acronym
def fxn(stng):
    # add first letter
    oupt = stng[0]

    # iterate over string
    for i in range(1, len(stng)):
        if stng[i - 1] == ' ':
            # add letter next to space
            oupt += stng[i]

    # uppercase oupt
    oupt = oupt.upper()
    return oupt


def noun2synset(noun):
    synset = wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*',
                                                                                    noun) else "'{}'".format(noun)
    noun = synset.split('.')[0]
    noun = noun.replace('_', ' ')

    if len(noun) > 15:
        noun = fxn(noun)

    return noun

def sort_dict(x):
    return dict(sorted(x.items(), key=lambda item: item[1], reverse=True))

def err_percent_dict(incorrect_dict, correct_dict, min_total=30):
    err_percents = {}
    for v in incorrect_dict.keys():
        v_correct = correct_dict[v] if v in correct_dict.keys() else 0
        v_err_percent = round(incorrect_dict[v] / (incorrect_dict[v] + v_correct), 2)
        total = incorrect_dict[v] + v_correct
        if total > min_total:
            err_percents[v] = v_err_percent

    err_percents = sort_dict(err_percents)
    return err_percents

def bar_plot_from_dict(items, title="", color="", max_items=-1):

    if max_items < 0:
        max_items = len(items.keys())

    x = list(items.keys())[:max_items]
    y = list(items.values())[:max_items]
    # Figure Size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Horizontal Bar Plot
    if color != "":
        ax.barh(x, y, color=color)
    else:
        ax.barh(x, y, color=(0.2, 0.4, 0.6, 0.6))

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=1)
    ax.yaxis.set_tick_params(pad=1)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.007, i.get_y() + 0.6,
                 str(round((i.get_width()), 2)),
                 fontsize=10,
                 color='grey')

    # Add Plot Title
    if title != "":
        ax.set_title(title, loc='left', )
    # Show Plot
    plt.show()

def pie_chart_from_dict(items:dict):
    fig1, ax1 = plt.subplots()
    ax1.pie(list(items.values()), explode=None, labels=list(items.keys()), autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

def process_n_nouns_count(nouns:dict, n_items=26):
    trimmed = {noun2synset(k): nouns[k] for idx, k in enumerate(nouns.keys()) if idx < n_items and k != 'blank'}
    return trimmed


def display_graphs_from_json(json_file):
    with open(json_file) as f:
        all_stats = json.load(f)

    # verbs_count = sort_dict(all_stats['verbs'])
    nouns_count = sort_dict(all_stats['nouns'])
    roles_count = sort_dict(all_stats['roles'])

    verb_errs = err_percent_dict(all_stats['verbs'], all_stats['verbs_correct'])
    bar_plot_from_dict(verb_errs, max_items=25)

    bar_plot_from_dict(roles_count, max_items=25, color='deepskyblue')

    # nouns_chart_dict = {k:nouns_count[k] for k in nouns_count.keys() if (nouns_count[k] >= 10) and (k != 'blank')}
    # others = 0
    # for k in nouns_count.keys():
    #     if nouns_count[k] < 10:
    #         others += nouns_count[k]
    # nouns_chart_dict['others'] = round(others / 4)

    processed_noun_count = process_n_nouns_count(nouns_count)
    bar_plot_from_dict(processed_noun_count, color='coral')





if __name__ == '__main__':
    logdir = Path('./flicker30k/pretrained/v7/')
    log_file = logdir / 'log_stats.txt'
    display_graphs_from_json(log_file)