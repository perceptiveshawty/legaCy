import pickle
import numpy as np
import csv
import spacy

from ext.luima_sbd.sbd import text2sentences
from ext.tmvm.extract_features import compute_grammatical_feats, get_clause_components
from ext.tmvm.split_units_ud import split_into_clauses, unit_from_sentence
from ext.lmtc.legalword2vec import LegalVectors
from mood import mood
from modality import modality

from utils import *

"""
Adapted code from:

https://gitlab.gwdg.de/tillmann.doenicke/disrpt2021-tmvm/-/tree/master
https://github.com/reglab/casehold/blob/main/multiple_choice/utils_multiple_choice.py
https://github.com/jsavelka/luima_sbd

Static word vectors from:
https://github.com/ashkonf/LeGloVe
https://archive.org/details/Law2Vec
"""


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
                if attribute_total > 0:
                    feature.append(tvmm[attribute][option] / attribute_total)
                else:
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


def extract_features(examples):

    nlp = spacy.load("en_core_web_lg")
    legalword2vec = LegalVectors()

    features = []
    for ex in examples:
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
    casehold = [CaseHOLDExample(line[0], line[1], line[2:7], int(line[-1])) for line in raw[1:] if line[-1] != ""]

    with open("data/casehold/datav2.pkl", "wb") as d:
        pickle.dump(casehold, d)

    features = extract_features(casehold)

    with open("checkpoints/fullv2.pkl", "wb") as d:
        pickle.dump(features, d)


if __name__ == "__main__":
    __main__()
