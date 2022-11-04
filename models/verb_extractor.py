from typing import List

from nltk.stem import SnowballStemmer
import spacy
from spacy import Language

from frame_semantic_transformer.data.tasks import FrameClassificationTask


class VerbExtractor:
    stemmer: SnowballStemmer
    spacy_nlp: Language
    be_verbs: List[str]

    def __init__(self):
        self.stemmer = SnowballStemmer(language='english')
        self.spacy_nlp = spacy.load("en_core_web_sm")

        self.be_verbs = ['am', 'are', 'is', 'was', 'were', 'been', 'being']

    def get_words_pos_list(self, s: str, words: List[str]):
        start = 0
        strlen = len(s)
        positions = []
        for w in words:
            spart = s[start:strlen]
            wpos = spart.find(w)
            if wpos != -1:
                positions.append(start + wpos)
                start = start + wpos + len(w)
        return positions

    def get_vbg_by_stemming(self, verb:str):
        vowels = ['a', 'e', 'i', 'o', 'u']

        root = self.stemmer.stem(verb)
        if root[-1] in vowels:
            root = root[:-1]

        framename = root + 'ing'
        return framename

    def get_swig_frames_from_verb(self, verb:str):
        framename = verb
        if not verb.endswith('ing'):
            doc_dep = self.spacy_nlp(verb)
            token = doc_dep[0]
            vbg = token._.inflect("VBG")
            if not vbg:
                vbg = self.get_vbg_by_stemming(verb)
            framename = vbg

        return framename

    def get_spacy_verbs(self, sentence:str):
        doc = self.spacy_nlp(sentence)
        spacy_verbs = [token.text for token in doc if token.pos_ == "VERB"]
        return spacy_verbs

    def get_verb_idx(self, sentence: str, with_frames:bool=True):
        verbs = self.get_spacy_verbs(sentence)
        verbs_pos = self.get_words_pos_list(sentence, verbs)

        if with_frames:
            frames_list = [self.get_swig_frames_from_verb(v) for v in verbs]
            return verbs_pos, frames_list

        return verbs_pos

def get_captions_from_tuple(captions:tuple) -> List[str]:
    tasks = []
    for c in captions:
        tasks.append(FrameClassificationTask(text=c[0], trigger_loc=c[1]))

    return [task.get_input() for task in tasks]