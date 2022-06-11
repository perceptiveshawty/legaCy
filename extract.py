from multiprocessing.sharedctypes import Value
import spacy
from ext.luima_sbd.sbd import text2sentences

from ext.tmvm.extract_features import compute_grammatical_feats, get_clause_components
from ext.tmvm.split_units_ud import split_into_clauses, unit_from_sentence

import numpy as np

from thinc.types import Floats1d, Dict, List, Tuple

from dataclasses import dataclass

from ext.lmtc.legalword2vec import LegalVectors
from mood import mood
from modality import modality

import csv

'''
Adapted code from:

https://gitlab.gwdg.de/tillmann.doenicke/disrpt2021-tmvm/-/tree/master
https://github.com/reglab/casehold/blob/main/multiple_choice/utils_multiple_choice.py
https://github.com/jsavelka/luima_sbd

Static word vectors from:
https://github.com/ashkonf/LeGloVe
https://archive.org/details/Law2Vec




'''


@dataclass(frozen=True)
class CaseHOLDInstance:
    """
    A single training/test example for the CaseHOLD multiple-choice question answering task.
    Args:
        example_id: Unique id for the example.
        context: str. The untokenized text of the first sequence (context of corresponding question).
        choices: list of str. Multiple choice's options.
        answer: int. Index of the correct answer in choices (zero-indexed)
    """

    example_id: int
    context: str
    choices: List[str]
    answer: int


@dataclass
class SymbolicFeatures:

    nouns: List[str]
    verbs: List[str]
    subjs: List[str]
    amods: List[str]
    phrases: List[Tuple[str, List[str]]]
    vector: Floats1d


@dataclass
class StructuralFeatures:

    verb_form: Floats1d  # ["Fin", "Inf", "Part"]
    tense: Floats1d  # ["Fut", "Past", "Pres"]
    aspect: Floats1d  # ["Imp", "Perf", "Prog"]
    mood: Floats1d  # ["Ind", "Cnd", "Sub"]
    voice: Floats1d  # ["Act", "Pass"]
    modality: Floats1d  # [Minimum, Mean, Maximum]
    vector: Floats1d


@dataclass
class CaseHOLDFeatures:
    example_id: int
    context_structural: StructuralFeatures
    context_symbolic: SymbolicFeatures
    choices_structural: List[StructuralFeatures]
    choices_symbolic: List[SymbolicFeatures]


# @dataclass
# class DifferentialCorpus:

#     appendix: Dict[int, List[str]]
#     vectors: Dict[int, Floats1d]


def caseHOLD_structural(examples):

    nlp = spacy.load("en_core_web_lg")
    legalword2vec = LegalVectors()

    features = []
    for ex in examples[:10000]:
        ans_strs, ans_syms = [], []
        ctx_str, ctx_sym = extract_structural_and_raw_symbolic(nlp, legalword2vec, ex.context)

        for ans in ex.choices:
            ans_str, ans_sym = extract_structural_and_raw_symbolic(nlp, legalword2vec, ans)

            ans_strs.append(ans_str)
            ans_syms.append(ans_sym)

        features.append(CaseHOLDFeatures(ex.example_id, ctx_str, ctx_sym, ans_strs, ans_syms))

    # to do: post process all examples to create corpus-wide vocabularies

    return features


def extract_structural_and_raw_symbolic(nlp, word2vec, text):

    NOUNS, VERBS, SUBJS, AMODS, PHRASES = (set(), set(), set(), set(), [])

    tvmm = {"VerbForm": dict.fromkeys(["Fin", "Inf", "Part"], 0), "Tense": dict.fromkeys(["Fut", "Past", "Pres"], 0), "Aspect": dict.fromkeys(["Imp", "Perf", "Prog"], 0), "Mood": dict.fromkeys(["Cnd", "Ind", "Sub"], 0), "Voice": dict.fromkeys(["Act", "Pass"], 0), "Modality": []}  # each sentence maps to a score in [-1, 1]

    sentences = text2sentences(text)

    for s in sentences:
        doc = nlp(s)
        tvmm["Mood"][mood(doc)] += 1
        tvmm["Modality"].append(modality(doc))

        for sent in doc.sents:
            sent_tokens = []
            for i, token in enumerate(sent):
                sent_tokens.append(
                    {"id": str(i + 1), "form": token.text, "lemma": token.lemma_, "upos": token.pos_, "tag": token.tag_, "feats": {k: v for k, v in [tuple(m.split("=")) for m in str(token.morph).split("|") if "=" in m]}, "head": str(list(sent).index(token.head) + 1), "deprel": token.dep_,}
                )

                if token.pos_ == "NOUN" and token.lemma_ in word2vec:
                    NOUNS.add(token.lemma_)

                if token.pos_ == "VERB" and token.lemma_ in word2vec:
                    VERBS.add(token.lemma_)

                if token.dep_ == "nsubj" and token.lemma_ in word2vec:
                    SUBJS.add(token.lemma_)

                if token.dep_ == "amod" and token.lemma_ in word2vec:
                    AMODS.add(token.lemma_)

            sentence = unit_from_sentence(sent_tokens, "eng")
            clauses = split_into_clauses(sentence)

            for clause in clauses:
                # split clause into composite verb, NPs and other elements
                composite_verb, NPs, _ = get_clause_components(clause)

                if len(composite_verb) > 0 and len(NPs) > 0:
                    phrase, has_verb, has_noun = set(), False, False
                    for conll_verb in composite_verb:
                        if conll_verb["form"].lower() in word2vec:
                            phrase.add(conll_verb["form"].lower())
                            has_verb = True

                    for np_unit in NPs:
                        for conll_noun in np_unit.tokens:
                            if conll_noun["form"].lower() in word2vec and conll_noun["upos"] == "NOUN":
                                phrase.add(conll_noun["form"].lower())
                                has_noun = True

                    if has_verb and has_noun and len(phrase) > 2 and len(phrase) < 7:
                        PHRASES.append(phrase)

                verb_feats = compute_grammatical_feats(composite_verb, "eng", reduce_infinite_forms=False)

                for feature in verb_feats:
                    if feature in ("VerbForm", "Tense", "Aspect", "Voice"):
                        frqkey = list(verb_feats[feature])[0]
                        tvmm[feature][frqkey] += 1

    feature = []
    for attribute in tvmm:
        if attribute in ("VerbForm", "Tense", "Aspect", "Voice", "Mood"):
            try:
                attribute_total = np.sum([np.sum(tvmm[attribute][j]) for j in tvmm[attribute]])
            except:
                attribute_total = 0
            for option in tvmm[attribute]:
                try:
                    feature.append(tvmm[attribute][option] / attribute_total)
                except:
                    feature.append(0)
        else:  # Modality - special case
            try:
                feature.append(np.min(tvmm[attribute]))
            except:
                feature.append(0)
            try:
                feature.append(np.mean(tvmm[attribute]))
            except:
                feature.append(0)
            try:
                feature.append(np.max(tvmm[attribute]))
            except:
                feature.append(0)

    return StructuralFeatures(feature[:3], feature[3:6], feature[6:9], feature[9:12], feature[12:14], feature[14:], feature), SymbolicFeatures(NOUNS, VERBS, SUBJS, AMODS, PHRASES, [])


# def __main__():

#     casehold = []
#     with open('data/casehold/data/casehold.csv', 'r', encoding='utf-8') as f:
#         raw = list(csv.reader(f))
#     casehold = [CaseHOLDInstance(line[0], line[1], line[2:7], line[-1]) for line in raw]

