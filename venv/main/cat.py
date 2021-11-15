from nltk.corpus import stopwords
import pprint
import re

# The argument is a list of strings
# Returns the string with all the possible categorizations by word subsets always containing the very last word and including the previous one subsequenty

def categorize_all(k):
    to_delete = stopwords.words('english')

    # kw_kw_kw_moreSpecific has the following structure:
    #  [ (kw_kw, kw_moreSpecific), ... ]
    kw_kw_kw_moreSpecific = list()


    ## PASO 1 __________________________________________________________________________________________________________

    # Maximum length of more specific sequences of postponed nouns
    max_n = None

    for kw in k:
        kw_kw = kw
        kw_kw = re.sub('^- ?', '', kw_kw)
        kw_kw = re.sub('"', '', kw_kw)
        kw_kw = re.sub('•', '', kw_kw)
        kw_kw = kw_kw.strip()
        kw_words = kw_kw.split()
        kw_nonstop = [w for w in kw_words if not w in to_delete]

        kw_moreSpecific = kw_kw
        kw_moreSpecific = re.sub('"', '', kw_moreSpecific)
        kw_moreSpecific = re.sub('"', '', kw_moreSpecific)
        kw_moreSpecific = re.sub('•', '', kw_moreSpecific)
        kw_moreSpecific = kw_moreSpecific.strip()

        ## PASO 2 __________________________________________________________________________________________________

        kw_moreSpecific_all = list()
        kw_moreSpecific_words = kw_moreSpecific.split()

        n = len(kw_moreSpecific_words)

        # Updating maximum length of more specific sequences of postponed nouns
        if max_n is None or max_n < n:
            max_n = n

        while n > 0:
            to_add = kw_moreSpecific_words[n-1:]
            kw_moreSpecific_all.append(to_add)
            n -= 1


        # All restrictions
        if len(kw_nonstop) > 0 and len([w for w in kw_words if len(w) <= 2]) == 0:

            for kw_moreSpecific_every in kw_moreSpecific_all:
                if (kw_kw, kw_moreSpecific_every) not in kw_kw_kw_moreSpecific:
                    kw_kw_kw_moreSpecific.append((kw_kw, " ".join(kw_moreSpecific_every)))



    ## PASO 3 __________________________________________________________________________________________________________

    ordered_list_tuples = list()
    for x in range(max_n):
        ordered_list_tuples.append([])

    for tupla in kw_kw_kw_moreSpecific:
        len_tupla = len(tupla[1].split())
        ordered_list_tuples[len_tupla - 1].append(tupla)


    ## PASO 4 __________________________________________________________________________________________________________

    # working_pairs has the following structure:
    # [ (list_n_menos_1, list_n) , (list_n_menos_2, list_n_menos_1) , ... ]
    working_pairs = list()

    for x in range(max_n - 1):
        working_pairs.append((ordered_list_tuples[-(x+1)], ordered_list_tuples[-(x+2)]))

    for n, pair in enumerate(working_pairs):

        if n == 0:
            tmp = dict()
            tmp_mostSpecific = dict()

            # For every preceding tuple (i.e., less specific), organize them into a dict
            for tupla_mostSpecicific in pair[0]:

                if tupla_mostSpecicific[1] in tmp_mostSpecific.keys() and tupla_mostSpecicific[0] not in tmp_mostSpecific[tupla_mostSpecicific[1]]:
                    tmp_mostSpecific[tupla_mostSpecicific[1]].append(tupla_mostSpecicific[0])
                else:
                    tmp_mostSpecific[tupla_mostSpecicific[1]] = [tupla_mostSpecicific[0]]

            # For every preceding tuple (i.e., less specific), organize them into a dict
            for tupla_ant in pair[1]:

                if tupla_ant[1] in tmp.keys() and tupla_ant[0] not in tmp[tupla_ant[1]]:
                    tmp[tupla_ant[1]].append(tupla_ant[0])
                else:
                    tmp[tupla_ant[1]] = [tupla_ant[0]]

            for cat_mostSpecific in tmp_mostSpecific:
                found = False
                for cat in list(tmp):

                    # If there is a less specific category to nest the categories analyzed
                    if cat_mostSpecific.split()[1:] == cat.split():
                        tmp[cat].append({cat_mostSpecific: tmp_mostSpecific[cat_mostSpecific]})

                        # Deleting duplicate values
                        for x in tmp_mostSpecific[cat_mostSpecific]:
                            if isinstance(x, str) and x in tmp[cat]:
                                tmp[cat].remove(x)

                        found = True

                if not found:
                    tmp[cat_mostSpecific] = tmp_mostSpecific[cat_mostSpecific]

        else:
            tmp_ant = dict()

            # For every preceding tuple (i.e., less specific), organize them into a dict
            for tupla_ant in pair[1]:

                if tupla_ant[1] in tmp_ant.keys():
                    tmp_ant[tupla_ant[1]].append(tupla_ant[0])
                else:
                    tmp_ant[tupla_ant[1]] = [tupla_ant[0]]

            for cat_moreSpecific in tmp:
                found = False
                for cat in list(tmp_ant):

                    len_generic_cat = -len(cat.split())

                    # If there is a less specific category to nest the analyzed categories
                    if cat.split() == cat_moreSpecific.split()[len_generic_cat:]:
                        tmp_ant[cat].append({cat_moreSpecific: tmp[cat_moreSpecific]})

                        # Deleting duplicate values
                        for x in getAllStrValues(tmp[cat_moreSpecific], []):
                            if isinstance(x, str) and x in tmp_ant[cat]:
                                tmp_ant[cat].remove(x)

                        found = True

                if not found:
                    tmp_ant[cat_moreSpecific] = tmp[cat_moreSpecific]

            tmp = tmp_ant

    return tmp_ant
