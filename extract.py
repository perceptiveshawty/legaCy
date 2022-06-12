from base64 import encode
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

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from umap import UMAP
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA


import csv

"""
Adapted code from:

https://gitlab.gwdg.de/tillmann.doenicke/disrpt2021-tmvm/-/tree/master
https://github.com/reglab/casehold/blob/main/multiple_choice/utils_multiple_choice.py
https://github.com/jsavelka/luima_sbd

Static word vectors from:
https://github.com/ashkonf/LeGloVe
https://archive.org/details/Law2Vec




"""


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


def L1_SOW(vectors):
    sum_ = np.sum(vectors)
    return sum_ / np.linalg.norm(sum)


def all_but_the_top(v, D):
    """
      All-but-the-Top: Simple and Effective Postprocessing for Word Representations
      https://arxiv.org/abs/1702.01417
      Arguments:
          :v: word vectors of shape (n_words, n_dimensions)
          :D: number of principal components to subtract
      """
    # 1. Subtract mean vector
    v_tilde = v - np.mean(v, axis=0)
    # 2. Compute the first `D` principal components
    #    on centered embedding vectors
    u = PCA(n_components=D).fit(v_tilde).components_  # [D, emb_size]
    # Subtract first `D` principal components
    # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
    return v_tilde - (v @ u.T @ u)


def cosine_sims(query, supers):
    norm = np.linalg.norm(query)
    all_norms = np.linalg.norm(supers, axis=1)
    dot_products = np.dot(supers, query)
    similarities = dot_products / (norm * all_norms)
    return similarities


def build_vocabulary(word2vec, features):
    vocabulary = {"N": {"word": [], "vec": []}, "V": {"word": [], "vec": []}, "S": {"word": [], "vec": []}, "A": {"word": [], "vec": []}}

    for chf in features:

        for n in chf.context_symbolic.nouns:
            if n not in vocabulary["N"]["word"]:
                vocabulary["N"]["word"].append(n)
                vocabulary["N"]["vec"].append(word2vec[n])

        for v in chf.context_symbolic.verbs:
            if v not in vocabulary["V"]["word"]:
                vocabulary["V"]["word"].append(v)
                vocabulary["V"]["vec"].append(word2vec[v])

        for s in chf.context_symbolic.subjs:
            if s not in vocabulary["S"]["word"]:
                vocabulary["S"]["word"].append(s)
                vocabulary["S"]["vec"].append(word2vec[s])

        for a in chf.context_symbolic.amods:
            if a not in vocabulary["A"]["word"]:
                vocabulary["A"]["word"].append(a)
                vocabulary["A"]["vec"].append(word2vec[a])

        for ans_sym in chf.choices_symbolic:
            for n in ans_sym.nouns:
                if n not in vocabulary["N"]["word"]:
                    vocabulary["N"]["word"].append(n)
                    vocabulary["N"]["vec"].append(word2vec[n])

            for v in ans_sym.verbs:
                if v not in vocabulary["V"]["word"]:
                    vocabulary["V"]["word"].append(v)
                    vocabulary["V"]["vec"].append(word2vec[v])

            for s in ans_sym.subjs:
                if s not in vocabulary["S"]["word"]:
                    vocabulary["S"]["word"].append(s)
                    vocabulary["S"]["vec"].append(word2vec[s])

            for a in ans_sym.amods:
                if a not in vocabulary["A"]["word"]:
                    vocabulary["A"]["word"].append(a)
                    vocabulary["A"]["vec"].append(word2vec[a])

    return vocabulary


def build_supervectors(word2vec, vocabulary):

    # NOUNS
    words, vecs = vocabulary["N"]["word"], all_but_the_top(np.vstack(vocabulary["N"]["vec"]), 1)
    constituents, supernouns = get_dense_clusters(word2vec, words, vecs)
    with open("knowledge/nouns/words.pkl", "wb") as d:
        pickle.dump(constituents, d)

    with open("knowledge/nouns/super.pkl", "wb") as d:
        pickle.dump(supernouns, d)

    # VERBS
    words, vecs = vocabulary["V"]["word"], all_but_the_top(np.vstack(vocabulary["V"]["vec"]), 1)
    constituents, superverbs = get_dense_clusters(word2vec, words, vecs)
    with open("knowledge/verbs/words.pkl", "wb") as d:
        pickle.dump(constituents, d)

    with open("knowledge/verbs/super.pkl", "wb") as d:
        pickle.dump(superverbs, d)

    # SUBJECTS
    words, vecs = vocabulary["S"]["word"], all_but_the_top(np.vstack(vocabulary["S"]["vec"]), 1)
    constituents, supersubjs = get_dense_clusters(word2vec, words, vecs)
    with open("knowledge/subjs/words.pkl", "wb") as d:
        pickle.dump(constituents, d)

    with open("knowledge/subjs/super.pkl", "wb") as d:
        pickle.dump(supersubjs, d)

    # ADJECTIVES
    words, vecs = vocabulary["A"]["word"], all_but_the_top(np.vstack(vocabulary["A"]["vec"]), 1)
    constituents, superadjs = get_dense_clusters(word2vec, words, vecs)
    with open("knowledge/adjs/words.pkl", "wb") as d:
        pickle.dump(constituents, d)

    with open("knowledge/adjs/super.pkl", "wb") as d:
        pickle.dump(superadjs, d)

    return list(supernouns.values()), list(superverbs.values()), list(supersubjs.values()), list(superadjs.values())


def get_dense_clusters(word2vec, words, vecs):
    x2d = UMAP(n_components=8, n_neighbors=3, min_dist=0.01, metric="cosine", random_state=444).fit_transform(vecs)
    optics = OPTICS(xi=0.1, min_cluster_size=0.01).fit(x2d)

    label2words, label2super = {}, {}
    for label, word in zip(optics.labels_, words):
        if label == -1:
            continue
        elif label not in label2words:
            label2words[label] = []
            label2super[label] = []

        label2words[label].append(word)
        label2super[label].append(word2vec[word])

    for label, subvecs in label2super.items():
        label2super[label] = np.mean(np.array(subvecs), axis=0)

    return label2words, label2super


def encode_symbolic(word2vec, supernouns, superverbs, supersubjs, superadjs, sf):

    nmax, vmax, smax, amax = np.zeros(len(supernouns)), np.zeros(len(superverbs)), np.zeros(len(supersubjs)), np.zeros(len(superadjs))
    nmin, vmin, smin, amin = np.ones(len(supernouns)), np.ones(len(superverbs)), np.ones(len(supersubjs)), np.ones(len(superadjs))

    for n in sf.nouns:
        sims = cosine_sims(word2vec[n], np.vstack(supernouns))
        nmax, nmin = np.maximum(nmax, sims), np.minimum(nmin, sims)

    for v in sf.verbs:
        sims = cosine_sims(word2vec[v], np.vstack(superverbs))
        vmax, vmin = np.maximum(vmax, sims), np.minimum(vmin, sims)

    for s in sf.subjs:
        sims = cosine_sims(word2vec[s], np.vstack(supersubjs))
        smax, smin = np.maximum(smax, sims), np.minimum(smin, sims)

    for a in sf.amods:
        sims = cosine_sims(word2vec[a], np.vstack(superadjs))
        amax, amin = np.maximum(amax, sims), np.minimum(amin, sims)

    sf.vector = np.concatenate([nmax, nmin, vmax, vmin, smax, smin, amax, amin])


def encode_structural_and_raw_symbolic(nlp, word2vec, text):

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


def extract(examples):

    nlp = spacy.load("en_core_web_lg")
    legalword2vec = LegalVectors()

    features = []
    for ex in examples[:1500]:
        ans_strs, ans_syms = [], []
        ctx_str, ctx_sym = encode_structural_and_raw_symbolic(nlp, legalword2vec, ex.context)

        for ans in ex.choices:
            ans_str, ans_sym = encode_structural_and_raw_symbolic(nlp, legalword2vec, ans)

            ans_strs.append(ans_str)
            ans_syms.append(ans_sym)

        features.append(CaseHOLDFeatures(ex.example_id, ctx_str, ctx_sym, ans_strs, ans_syms))

    vocabulary = build_vocabulary(legalword2vec, features)
    supernn, supervb, supersj, superad = build_supervectors(legalword2vec, vocabulary)

    for chf in features:
        encode_symbolic(legalword2vec, supernn, supervb, supersj, superad, chf.context_symbolic)
        for ans_sym in chf.choices_symbolic:
            encode_symbolic(legalword2vec, supernn, supervb, supersj, superad, ans_sym)

    return features


def __main__():

    casehold = []
    with open("data/casehold/data/casehold.csv", "r", encoding="utf-8") as f:
        raw = list(csv.reader(f))
    casehold = [CaseHOLDInstance(line[0], line[1], line[2:7], line[-1]) for line in raw[1:]]

    features = extract(casehold)

    with open("checkpoints/fullv1.pkl", "wb") as d:
        pickle.dump(features, d)


if __name__ == "__main__":
    __main__()
