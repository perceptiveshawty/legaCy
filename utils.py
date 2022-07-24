from dataclasses import dataclass
from thinc.types import Floats1d, List, Tuple

import pickle
import numpy as np

from umap import UMAP
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class CaseHOLDExample:
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
    """
    The computed symbolic features for a given instance, separated by grammatical roles.
    """

    nouns: List[str]
    verbs: List[str]
    subjs: List[str]
    amods: List[str]
    phrases: List[Tuple[str, List[str]]]
    vector: Floats1d


@dataclass
class StructuralFeatures:
    """
    The computed structural features for a given instance.
    """

    verb_form: Floats1d  # ["Fin", "Inf", "Part"]
    tense: Floats1d  # ["Fut", "Past", "Pres"]
    aspect: Floats1d  # ["Imp", "Perf", "Prog"]
    mood: Floats1d  # ["Ind", "Cnd", "Sub"]
    voice: Floats1d  # ["Act", "Pass"]
    modality: Floats1d  # [Minimum, Mean, Maximum]
    vector: Floats1d


@dataclass
class CaseHOLDFeatures:
    """
    Wrapper for structural and symbolic features associated with a single example.
    """

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

    id2pos, supervals = {"N": "nouns", "V": "verbs", "S": "subjs", "A": "adjs"}, []

    for key, name in id2pos.items():
        words, vecs = vocabulary[key]["word"], all_but_the_top(np.vstack(vocabulary[key]["vec"]), 1)
        constituents, supers = get_dense_clusters(word2vec, words, vecs)
        with open(f"knowledge/{name}/words.pkl", "wb") as d:
            pickle.dump(constituents, d)

        with open(f"knowledge/{name}/super.pkl", "wb") as d:
            pickle.dump(supers, d)

        supervals.append(list(supers.values()))

    return supervals

    # # NOUNS
    # words, vecs = vocabulary["N"]["word"], all_but_the_top(np.vstack(vocabulary["N"]["vec"]), 1)
    # constituents, supernouns = get_dense_clusters(word2vec, words, vecs)
    # with open("knowledge/nouns/words.pkl", "wb") as d:
    #     pickle.dump(constituents, d)

    # with open("knowledge/nouns/super.pkl", "wb") as d:
    #     pickle.dump(supernouns, d)

    # # VERBS
    # words, vecs = vocabulary["V"]["word"], all_but_the_top(np.vstack(vocabulary["V"]["vec"]), 1)
    # constituents, superverbs = get_dense_clusters(word2vec, words, vecs)
    # with open("knowledge/verbs/words.pkl", "wb") as d:
    #     pickle.dump(constituents, d)

    # with open("knowledge/verbs/super.pkl", "wb") as d:
    #     pickle.dump(superverbs, d)

    # # SUBJECTS
    # words, vecs = vocabulary["S"]["word"], all_but_the_top(np.vstack(vocabulary["S"]["vec"]), 1)
    # constituents, supersubjs = get_dense_clusters(word2vec, words, vecs)
    # with open("knowledge/subjs/words.pkl", "wb") as d:
    #     pickle.dump(constituents, d)

    # with open("knowledge/subjs/super.pkl", "wb") as d:
    #     pickle.dump(supersubjs, d)

    # # ADJECTIVES
    # words, vecs = vocabulary["A"]["word"], all_but_the_top(np.vstack(vocabulary["A"]["vec"]), 1)
    # constituents, superadjs = get_dense_clusters(word2vec, words, vecs)
    # with open("knowledge/adjs/words.pkl", "wb") as d:
    #     pickle.dump(constituents, d)

    # with open("knowledge/adjs/super.pkl", "wb") as d:
    #     pickle.dump(superadjs, d)

    # return list(supernouns.values()), list(superverbs.values()), list(supersubjs.values()), list(superadjs.values())


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
