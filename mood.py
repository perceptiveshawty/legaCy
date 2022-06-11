""" 
    Written by http://www.clips.ua.ac.be/pages/pattern
    Adapted for legaCy
"""


def find(function, list):
    """ Returns the first item in the list for which function(item) is True, None otherwise.
    """
    for item in list:
        if function(item):
            return item


def conditional(doc, predictive=True, **kwargs):
    """ The conditional mood is used to talk about possible or imaginary situations.
        It is marked by the infinitive form of the verb, preceded by would/could/should:
        "we should be going", "we could have stayed longer".
        With predictive=False, sentences with will/shall need an explicit if/when/once-clause:
        - "I will help you" => predictive.
        - "I will help you if you pay me" => speculative.
        Sentences with can/may always need an explicit if-clause.

        Adapted from http://www.clips.ua.ac.be/pages/pattern
    """
    S, tokens = doc.text, [t for t in doc]

    if "?" in S:
        return False

    i = find(lambda i: tokens[i].text.lower() == "were", range(len(tokens)))
    if i is None:
        i = 0

    if i > 0 and (tokens[i - 1].text.lower() in ("i", "it", "he", "she") or tokens[i - 1].tag_ == "NN"):
        # "As if it were summer already." => subjunctive (wish).
        return False

    for i, w in enumerate(tokens):
        if w.tag_ == "MD":
            if w.text.lower() == "ought" and i < len(tokens) - 1 and tokens[i + 1].text.lower() == "to":
                # "I ought to help you."
                return True
            if w.text.lower() in ("would", "should", "'d", "could", "might"):
                # "I could help you."
                return True
            if w.text.lower() in ("will", "shall", "'ll") and i > 0 and tokens[i - 1].text.lower() == "you":
                # "You will help me." => imperative.
                return False
            if w.text.lower() in ("will", "shall", "'ll") and predictive:
                # "I will help you." => predictive.
                return True
            if w.text.lower() in ("will", "shall", "'ll", "can", "may"):
                # "I will help you when I get back." => speculative.
                r = S.lower().rstrip(" .!")
                for cc in ("if", "when", "once", "as soon as", "assuming", "provided that", "given that"):
                    if cc + " " in r:
                        return True

    return False


subjunctive1 = ["advise", "ask", "command", "demand", "desire", "insist", "propose", "recommend", "request", "suggest", "urge"]
subjunctive2 = ["best", "crucial", "desirable", "essential", "imperative", "important", "recommended", "urgent", "vital"]

for w in list(subjunctive1):  # Inflect.
    subjunctive1.append(w + "s")
    subjunctive1.append(w.rstrip("e") + "ed")


def subjunctive(doc, classical=True, **kwargs):
    """ The subjunctive mood is a classical mood used to express a wish, judgment or opinion.
        It is marked by the verb wish/were, or infinitive form of a verb
        preceded by an "it is"-statement:
        "It is recommended that he bring his own computer."

        Adapted from http://www.clips.ua.ac.be/pages/pattern
    """
    S, tokens = doc.text, [t for t in doc]

    if "?" in S:
        return False

    for i, w in enumerate(tokens):
        b = False
        if w.tag_.startswith("VB"):
            if w.text.lower().startswith("wish"):
                # "I wish I knew."
                return True
            if w.text.lower() == "hope" and i > 0 and tokens[i - 1].text.lower() in ("i", "we"):
                # "I hope ..."
                return True
            if w.text.lower() == "were" and i > 0 and (tokens[i - 1].text.lower() in ("i", "it", "he", "she") or tokens[i - 1].tag_ == "NN"):
                # "It is as though she were here." => counterfactual.
                return True
            if w.text.lower() in subjunctive1:
                # "I propose that you be on time."
                b = True
            elif w.text.lower() == "is" and 0 < i < len(tokens) - 1 and tokens[i - 1].text.lower() == "it" and tokens[i + 1].text.lower() in subjunctive2:
                # "It is important that you be there." => but you aren't (yet).
                b = True
            elif w.text.lower() == "is" and 0 < i < len(tokens) - 3 and tokens[i - 1].text.lower() == "it" and tokens[i + 2].text.lower() in ("good", "bad") and tokens[i + 3].text.lower() == "idea":
                # "It is a good idea that you be there."
                b = True
        if b:
            # With classical=False, "It is important that you are there." passes.
            # This is actually an informal error: it states a fact, not a wish.
            v = find(lambda w: w.tag_.startswith("VB"), tokens[i + 1 :])
            if v and classical is True and v and v.tag_ == "VB":
                return True
            if v and classical is False:
                return True
    return False


def mood(doc):
    if conditional(doc):
        return "Cnd"
    elif subjunctive(doc):
        return "Sub"
    else:
        return "Ind"
