""" 
    Written by http://www.clips.ua.ac.be/pages/pattern
    Adapted for legaCy
"""


def d(*args):
    return dict.fromkeys(args, True)


AUXILLARY = {"be": ["be", "am", "m", "are", "is", "being", "was", "were" "been"], "can": ["can", "ca", "could"], "dare": ["dare", "dares", "daring", "dared"], "do": ["do", "does", "doing", "did", "done"], "have": ["have", "ve", "has", "having", "had"], "may": ["may", "might"], "must": ["must"], "need": ["need", "needs", "needing", "needed"], "ought": ["ought"], "shall": ["shall", "sha"], "will": ["will", "ll", "wo", "willing", "would", "d"]}

MODIFIERS = ("fully", "highly", "most", "much", "strongly", "very")

EPISTEMIC = "epistemic"  # Expresses degree of possiblity.

# -1.00 = NEGATIVE
# -0.75 = NEGATIVE, with slight doubts
# -0.50 = NEGATIVE, with doubts
# -0.25 = NEUTRAL, slightly negative
# +0.00 = NEUTRAL
# +0.25 = NEUTRAL, slightly positive
# +0.50 = POSITIVE, with doubts
# +0.75 = POSITIVE, with slight doubts
# +1.00 = POSITIVE

epistemic_MD = {  # would => could => can => should => shall => will => must
    -1.00: d(),
    -0.75: d(),
    -0.50: d("would"),
    -0.25: d("could", "dare", "might"),
    0.00: d("can", "ca", "may"),
    +0.25: d("ought", "should"),
    +0.50: d("shall", "sha"),
    +0.75: d("will", "'ll", "wo"),
    +1.00: d("have", "has", "must", "need"),
}

epistemic_VB = {  # wish => feel => believe => seem => think => know => prove + THAT
    -1.00: d(),
    -0.75: d(),
    -0.50: d("dispute", "disputed", "doubt", "question"),
    -0.25: d("hope", "want", "wish"),
    0.00: d("guess", "imagine", "seek"),
    +0.25: d("appear", "bet", "feel", "hear", "rumor", "rumour", "say", "said", "seem", "seemed", "sense", "speculate", "suspect", "suppose", "wager"),
    +0.50: d("allude", "anticipate", "assume", "claim", "claimed", "believe", "believed", "conjecture", "consider", "considered", "decide", "expect", "find", "found", "hypothesize", "imply", "indicate", "infer", "postulate", "predict", "presume", "propose", "report", "reported", "suggest", "suggested", "tend", "think", "thought"),
    +0.75: d("know", "known", "look", "see", "show", "shown"),
    +1.00: d("certify", "demonstrate", "prove", "proven", "verify"),
}

epistemic_RB = {  # unlikely => supposedly => maybe => probably => usually => clearly => definitely
    -1.00: d("impossibly"),
    -0.75: d("hardly"),
    -0.50: d("presumptively", "rarely", "scarcely", "seldomly", "uncertainly", "unlikely"),
    -0.25: d("almost", "allegedly", "debatably", "nearly", "presumably", "purportedly", "reportedly", "reputedly", "rumoredly", "rumouredly", "supposedly"),
    0.00: d("barely", "hypothetically", "maybe", "occasionally", "perhaps", "possibly", "putatively", "sometimes", "sporadically", "traditionally", "widely"),
    +0.25: d("admittedly", "apparently", "arguably", "believably", "conceivably", "feasibly", "fairly", "hopefully", "likely", "ostensibly", "potentially", "probably", "quite", "seemingly"),
    +0.50: d("commonly", "credibly", "defendably", "defensibly", "effectively", "frequently", "generally", "largely", "mostly", "normally", "noticeably", "often", "plausibly", "reasonably", "regularly", "relatively", "typically", "usually"),
    +0.75: d("assuredly", "certainly", "clearly", "doubtless", "evidently", "evitably", "manifestly", "necessarily", "nevertheless", "observably", "ostensively", "patently", "plainly", "positively", "really", "surely", "truly", "undoubtably", "undoubtedly", "verifiably"),
    +1.00: d("absolutely", "always", "definitely", "incontestably", "indisputably", "indubitably", "ineluctably", "inescapably", "inevitably", "invariably", "obviously", "unarguably", "unavoidably", "undeniably", "unquestionably"),
}

epistemic_JJ = {
    -1.00: d("absurd", "prepostoreous", "ridiculous"),
    -0.75: d("inconceivable", "unthinkable"),
    -0.50: d("misleading", "scant", "unlikely", "unreliable"),
    -0.25: d("customer-centric", "doubtful", "ever", "ill-defined, " "inadequate", "late", "uncertain", "unclear", "unrealistic", "unspecified", "unsure", "wild"),
    0.00: d("dynamic", "possible", "unknown"),
    +0.25: d("according", "creative", "likely", "local", "innovative", "interesting", "potential", "probable", "several", "some", "talented", "viable"),
    +0.50: d("certain", "generally", "many", "notable", "numerous", "performance-oriented", "promising", "putative", "well-known"),
    +0.75: d("concrete", "credible", "famous", "important", "major", "necessary", "original", "positive", "significant", "real", "robust", "substantial", "sure"),
    +1.00: d("confirmed", "definite", "prime", "undisputable"),
}

epistemic_NN = {
    -1.00: d("fantasy", "fiction", "lie", "myth", "nonsense"),
    -0.75: d("controversy"),
    -0.50: d("criticism", "debate", "doubt"),
    -0.25: d("belief", "chance", "faith", "luck", "perception", "speculation"),
    0.00: d("challenge", "guess", "feeling", "hunch", "opinion", "possibility", "question"),
    +0.25: d("assumption", "expectation", "hypothesis", "notion", "others", "team"),
    +0.50: d("example", "proces", "theory"),
    +0.75: d("conclusion", "data", "evidence", "majority", "proof", "symptom", "symptoms"),
    +1.00: d("fact", "truth", "power"),
}

epistemic_CC_DT_IN = {0.00: d("either", "whether"), +0.25: d("however", "some"), +1.00: d("despite")}

epistemic_PRP = {+0.25: d("I", "my"), +0.50: d("our"), +0.75: d("we")}

epistemic_weaseling = {
    -0.75: d("popular belief"),
    -0.50: d("but that", "but this", "have sought", "might have", "seems to"),
    -0.25: d("may also", "may be", "may have", "may have been", "some have", "sort of"),
    +0.00: d("been argued", "believed to", "considered to", "claimed to", "is considered", "is possible", "overall solutions", "regarded as", "said to"),
    +0.25: d("a number of", "in some", "one of", "some of", "many modern", "many people", "most people", "some people", "some cases", "some studies", "scientists", "researchers"),
    +0.50: d("in several", "is likely", "many of", "many other", "of many", "of the most", "such as", "several reasons", "several studies", "several universities", "wide range"),
    +0.75: d("almost always", "and many", "and some", "around the world", "by many", "in many", "in order to", "most likely"),
    +1.00: d("i.e.", "'s most", "of course", "There are", "without doubt"),
}


def modality(doc, type=EPISTEMIC):
    """ Returns the sentence's modality as a weight between -1.0 and +1.0.
        Currently, the only type implemented is EPISTEMIC.
        Epistemic modality is used to express possibility (i.e. how truthful is what is being said).

        Adapted from http://www.clips.ua.ac.be/pages/pattern
    """

    S, tokens, n, m = doc.text, [t for t in doc], 0.0, 0

    if type == EPISTEMIC:
        r = S.rstrip(" .!")

        for k, v in epistemic_weaseling.items():
            for phrase in v:
                if phrase in r:
                    n += k
                    m += 2

        for i, w in enumerate(tokens):
            for type, dict, weight in (("MD", epistemic_MD, 4), ("VB", epistemic_VB, 2), ("RB", epistemic_RB, 2), ("JJ", epistemic_JJ, 1), ("NN", epistemic_NN, 1), ("CC", epistemic_CC_DT_IN, 1), ("DT", epistemic_CC_DT_IN, 1), ("IN", epistemic_CC_DT_IN, 1), ("PRP", epistemic_PRP, 1), ("PRP$", epistemic_PRP, 1), ("WP", epistemic_PRP, 1)):

                # "likely" => weight 1, "very likely" => weight 2
                if i > 0 and tokens[i - 1].text.lower() in MODIFIERS:
                    weight += 1
                # likely" => score 0.25 (neutral inclining towards positive).
                if w.tag_ and w.tag_.startswith(type):
                    for k, v in dict.items():
                        # Prefer lemmata.
                        if (w.lemma_.lower() or w.text.lower()) in v:
                            # Reverse score for negated terms.
                            if i > 0 and tokens[i - 1].text.lower() in ("not", "n't", "never", "without"):
                                k = -k * 0.5
                            n += weight * k
                            m += weight
                            break
            # Numbers, citations, explanations make the sentence more factual.
            if w.tag_ in ("CD", '"', "'", ":", "("):
                n += 0.75
                m += 1
    if m == 0:
        return 1.0  # No modal verbs/adverbs used, so statement must be true.
    return max(-1.0, min(n / (m or 1), +1.0))


def uncertain(sentence, threshold=0.5):
    return modality(sentence) <= threshold
