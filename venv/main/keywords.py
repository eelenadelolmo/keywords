from nltk.corpus import stopwords
import textacy
import en_core_web_lg

all = 'corpus/Indiana_reports/indiana_reports_findings_impressions.txt'

## Required installations:

# pip install nltk
# pip install spacy==2.3.1
# download the tar.gz file from: https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-2.3.1
# pip install en_core_web_lg-2.3.1.tar.gz
# pip install textacy==0.10.0

# Add the following function to textaxy/extract.py

"""

Extract sequences of consecutive tokens from a spacy-parsed doc whose
part-of-speech tags match the specified regex pattern.

Args:
    doc
    pattern: Pattern of consecutive POS tags whose corresponding words
        are to be extracted, inspired by the regex patterns used in NLTK's
        `nltk.chunk.regexp`. Tags are uppercase, from the universal tag set;
        delimited by < and >, which are basically converted to parentheses
        with spaces as needed to correctly extract matching word sequences;
        white space in the input doesn't matter.

        Examples (see ``constants.POS_REGEX_PATTERNS``):

        * noun phrase: r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
        * compound nouns: r'<NOUN>+'
        * verb phrase: r'<VERB>?<ADV>*<VERB>+'
        * prepositional phrase: r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'

Yields:
    Next span of consecutive tokens from ``doc`` whose parts-of-speech match ``pattern``,
    in order of appearance

Warning:
    *DEPRECATED!* For similar but more powerful and performant functionality,
    use :func:`textacy.extract.matches()` instead.

def pos_regex_matches_tag(doc: Union[Doc, Span], pattern: str) -> Iterable[Span]:


    # standardize and transform the regular expression pattern...
    pattern = re.sub(r"\s", "", pattern)
    pattern = re.sub(r"<([A-Z]+)\|([A-Z]+)>", r"( (\1|\2))", pattern)
    pattern = re.sub(r"<([A-Z]+)>", r"( \1)", pattern)

    tags = " " + " ".join(tok.tag_ for tok in doc)

    for m in re.finditer(pattern, tags):
        yield doc[tags[0 : m.start()].count(" ") : tags[0 : m.end()].count(" ")]

"""

# changed in keywords/venv/lib/python3.8/site-packages/spacy/language.py
# nlp.max_length = 10000000000


# For avoiding overlapping matches with overlapping regex and computed to generate the keywords
# Get a list of noun phrases and deletes the overlapping shorter ones
def delete_overlapping(np_list):
    for i, elem_i in enumerate(np_list):
        for j in range(i):
            elem_j = np_list[j]
            if not elem_j:
                continue
            if elem_j.start <= elem_i.start and elem_j.end >= elem_i.end:
                # elem_i inside elem_j
                np_list[i] = None
                break
            elif elem_i.start <= elem_j.start and elem_i.end >= elem_j.end:
                # elem_j inside elem_i
                np_list[j] = None
            elif elem_i.end > elem_j.start and elem_i.start < elem_j.end:
                continue
                # raise ValueError('partial overlap?')
    return [elem for elem in np_list if elem]

## Computing frequencies

def absolute_freq_calc(kw, text):
    return (text.count(kw.lower()),)

def no_stopwords_freq_calc(kw, text):
    to_delete = stopwords.words('english')
    kw_words = kw.split()
    kw_nonstop = [w for w in kw_words if not w in to_delete]

    freq = 0
    if len(kw_nonstop) > 0:
        for k_n in kw_nonstop:
            freq += text.count(k_n.lower())
        freq = int(freq / len(kw_nonstop))

    return (freq,)

def calculate_word_scores_RAKE(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = phrase.split()
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)

    return (word_frequency, word_degree, word_score)

def freq_calc_RAKE(phrase, w_s):
    word_list = phrase.split()
    candidate_score = 0
    for word in word_list:
        candidate_score += w_s[word]
    return candidate_score

def degree_calc_RAKE(phrase, w_d):
    word_list = phrase.split()
    candidate_score = 0
    for word in word_list:
        candidate_score += w_d[word]
    return candidate_score

def ratio_calc_RAKE(phrase, w_r):
    word_list = phrase.split()
    candidate_score = 0
    for word in word_list:
        candidate_score += w_r[word]
    return candidate_score


# The argument os the path to a text file
# Returns a list of keywords based on PoS patterns:
def extract_keywords(dir):

    # Pattern based on fine-grained tag (must be searched with pos_regex_matches_tag)
    pattern_np = '<CD>? ((<RB>|<RBR>|<RBS>)? (<RB>|<RBR>|<RBS>)? (<JJ>|<JJR>|<JJS>|<VBG>|<VBN>)+ <CC>?)* (<NNP>|<NN>|<NNS>)+ ((<RB>|<RBR>|<RBS>)? (<RB>|<RBR>|<RBS>)? <VBN> <CC>?)*'

    # Pattern based on coarse tag because the tag <,> appears to be not matched (must be searched with pos_regex_matches)
    pattern_np_punct = '<NUM>? (<ADV>? <ADV>? (<ADJ>|<VERB>)+ (<PUNCT> (<ADJ>|<VERB>))? <CCONJ>?)* (<NOUN>|<PROPN>)+ (<ADV>? <ADV>? <VERB> (<PUNCT> <VERB>)? <CCONJ>?)*'

    # Pattern based on fine-grained tag (must be searched with pos_regex_matches_tag)
    pattern_np_pp = '<CD>? ((<RB>|<RBR>|<RBS>)? (<RB>|<RBR>|<RBS>)? (<JJ>|<JJR>|<JJS>|<VBG>|<VBN>)+ <CC>?)* (<NNP>|<NN>|<NNS>)+ ((<RB>|<RBR>|<RBS>)? (<RB>|<RBR>|<RBS>)? <VBN> <CC>?)* <IN> (<NNP>|<NN>|<NNS>)+'

    # Pattern based on coarse tag because the tag <,> appears to be not matched (must be searched with pos_regex_matches)
    pattern_np_punct_pp = '<NUM>? (<ADV>? <ADV>? (<ADJ>|<VERB>)+ (<PUNCT> (<ADJ>|<VERB>))? <CCONJ>?)* (<NOUN>|<PROPN>)+ (<ADV>? <ADV>? <VERB> (<PUNCT> <VERB>)? <CCONJ>?)* <ADP> (<NOUN>|<PROPN>)+'

    # Pattern based on fine-grained tag (must be searched with pos_regex_matches_tag)
    pattern_np_jj = '((<JJ>|<JJR>|<JJS>|<VBG>|<VBN>)+ <CC>?)* (<NN>|<NNP>|<NNS>)+ (<VBN> <CC>?)*'

    # Pattern based on coarse tag because the tag <,> appears to be not matched (must be searched with pos_regex_matches)
    pattern_np_jj_punct = '((<ADJ>|>VERB>)+ (<PUNCT> <ADJ>|<VERB>)? <CCONJ>?)* (<NOUN>|<PROPN>)+ (<VERB> (<PUNCT> <VERB>)? <CCONJ>?)*'


    with open(dir) as f:
        text = f.read().lower()

    # paras_list = text.split('\n\n\n')
    paras_list = text.split('\n')

    # List of noun phrases
    salida_np = list()

    # List of noun phrases plus bare preposition phrases
    salida_np_pp = list()

    # List of nouns plus adjective complements
    salida_np_jj = list()

    # For performance reasons
    for para in paras_list:
        doc = textacy.make_spacy_doc(para, lang=u'en_core_web_lg')

        matches_jj = list(textacy.extract.pos_regex_matches_tag(doc, pattern_np_jj))
        salida_np_jj.extend(matches_jj)

        # Comment in order not to match more than two coordinates coordinations
        matches_punct_jj = list(textacy.extract.pos_regex_matches(doc, pattern_np_jj_punct))
        matches_punct_jj_ok = list()

        # Avoid regex matches including punctuation different from comma (coodinations without commas are also matched, as they are already in matches_punct_jj, matched by pattern_np_jj)
        for i, e in enumerate(matches_punct_jj):
            if ',' in e.lower_:
                matches_punct_jj_ok.append(e)
        salida_np_jj.extend(matches_punct_jj_ok)


    ## Cleaning up the keywords list
    matches_clean_jj = list()
    for e in salida_np_jj:

        to_add = list()
        has_coord = False

        list_nn = list()
        root = None

        span_list = list(e)
        reversed_e = span_list[::-1]

        for t in e:

            if t.pos_ == 'NOUN':
                list_nn.append(t)
            if len(list_nn) > 0:
                root = list_nn[-1]
                root_complete = [x for x in list(root.subtree) if x.text in e.text.split()]

            if t.pos_ == 'CCONJ' or t.pos_ == 'PUNCT':
                has_coord = True

        root_all_nns = list()
        if root:
            for t in reversed_e:
                if t.pos_ != 'NOUN':
                    break
                if t.pos_ == 'NOUN':
                    root_all_nns.append(t)
            if len(root_all_nns) <= 1:
                root_all_nns = None

        if has_coord:

            if root:

                if ',' not in e.text.replace(',', ' ,').split():
                    conj = [x for x in e if x.dep_ == 'cc']
                    if len(conj) > 0:
                        conj = conj[0]
                        first_head = conj.head
                        second_head = [x for x in e if x.dep_ == 'conj']

                        if len(second_head) > 0:
                            second_head = second_head[0]
                            second_complete = [x for x in list(second_head.subtree) if x.text in e.text.split()]
                            first_complete = [x for x in list(first_head.subtree) if
                                              x not in second_complete and x.pos_ != 'CCONJ' and x.text in e.text.split()]
                            root_complete_no_coords = [x for x in list(root.subtree) if
                                                       x.text in e.text.split() and x not in first_complete + second_complete and x.pos_ != 'CCONJ']

                            if len(root_complete_no_coords) > 0 and len([x.text for x in root_complete_no_coords]) == len(
                                    set([x.text for x in root_complete_no_coords])):
                                too_add_1 = " ".join([i.text for i in first_complete]) + " " + " ".join(
                                    [i.text for i in root_complete_no_coords])
                                too_add_2 = " ".join([i.text for i in second_complete]) + " " + " ".join(
                                    [i.text for i in root_complete_no_coords])

                                # Keeping lemmas and PoS tags for every keyword
                                lemmas = str()
                                pos = str()
                                for t in e:
                                    lemmas += t.lemma_ + ' '
                                    pos += t.pos_ + ' '
                                lemmas = lemmas[:-1]
                                pos = pos[:-1]

                                to_add.append(((too_add_1, lemmas, pos), root))
                                to_add.append(((too_add_2, lemmas, pos), root))

                if ',' in e.text:

                    list_childs = [x for x in list(root.children) if
                                   x.text in e.text.replace(',', ' ,').split() and x.pos_ == 'ADJ']
                    root_complete_no_coords = [x for x in list(root.subtree) if x.text in e.text.replace(',',
                                                                                                         ' ,').split() and x not in list_childs and x.pos_ != 'PUNCT' and x.pos_ != 'CCONJ']
                    # Keeping lemmas and PoS tags for every keyword
                    lemmas = str()
                    pos = str()
                    for t in e:
                        lemmas += t.lemma_ + ' '
                        pos += t.pos_ + ' '
                    lemmas = lemmas[:-1]
                    pos = pos[:-1]

                    for yuxt in list_childs:
                        to_add_yuxt = " ".join([i.text for i in list(yuxt.subtree) if
                                                i.text in e.text and i not in root_complete_no_coords and i.pos_ != 'PUNCT' and i.pos_ != 'CCONJ']) + " " + " ".join(
                            [i.text for i in root_complete_no_coords])
                        to_add.append(((to_add_yuxt, lemmas, pos), root))


        else:
            if root:

                # Keeping lemmas and PoS tags for every keyword
                lemmas = str()
                pos = str()
                for t in e:
                    lemmas += t.lemma_ + ' '
                    pos += t.pos_ + ' '
                lemmas = lemmas[:-1]
                pos = pos[:-1]

                keyword = e.text
                for t in e:
                    if (((t.pos_ == 'ADV' or t.pos_ == 'CCONJ') and t.head.tag_ not in ['JJ', 'JJR', 'JJS', 'VBN', 'VBG']) or t.lemma_ == '•'):
                        keyword.replace(e.text.replace(t.text, "").replace('however', ""))

                if root_all_nns is not None and len(root_all_nns) > 1:
                    to_add.append(((keyword, lemmas, pos), root, root_all_nns))
                if root_all_nns is None or len(root_all_nns) <= 1:
                    to_add.append(((keyword, lemmas, pos), root))

        for tup in to_add:
            if tup[0][0] not in [x[0][0] for x in matches_clean_jj]:
                matches_clean_jj.append(tup)


    output = list()
    keywords = list()

    for tup in matches_clean_jj:
        output.append(tup[0])
        keywords.append(tup[0][0])

    RAKE_scores = calculate_word_scores_RAKE(keywords)

    for n, t in enumerate(output):
        output[n] = t + absolute_freq_calc(t[0], text) + no_stopwords_freq_calc(t[0], text) + (freq_calc_RAKE(t[0], RAKE_scores[0]),) + (degree_calc_RAKE(t[0], RAKE_scores[1]),) + (ratio_calc_RAKE(t[0], RAKE_scores[2]),)

    return [("words", "lemmas", "pos tags", "absolute freq", "no_stopwords absolute freq", "words absolute freq (RAKE)", "words degree (RAKE)", "words ratio (RAKE)")] + output


print(extract_keywords(all))
