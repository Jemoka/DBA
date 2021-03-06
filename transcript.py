# os utilities
import os
# pathing
import pathlib
# regex
import re
# glob
import glob
# open pd
import pandas as pd
# defaultdict
from collections import defaultdict
# import pickle
import pickle

from pandas.core.generic import _align_as_utc
from seaborn import relational

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob.glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, pathlib.Path(file_path).name)

# path of CLAN
CLAN_PATH=""

# file to check
DATADIR_AD="/Users/houliu/Documents/Projects/DBA/data/raw/pitt-07-24/dementia"
OUTDIR_AD="/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-24/dementia"

DATADIR_C="/Users/houliu/Documents/Projects/DBA/data/raw/pitt-07-24/control"
OUTDIR_C="/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-24/control"

TOKENDIR="/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-24/tokens.bin"

# tier to read
READ="*PAR"

# create a defaultdict to encode syntax features
tokens = defaultdict(lambda:len(tokens))


def do(DATADIR, OUTDIR):

    # get output
    files = globase(DATADIR, "*.cha")

    # for each file
    for checkfile in files:

        # run flo on the file
        CMD = f"{os.path.join(CLAN_PATH, 'flo +t%wor +t%xwor +t%mor +t%gra +t@ID')} {checkfile} >/dev/null 2>&1"
        # run!
        os.system(CMD)

        # path to result
        result_path = checkfile.replace("cha", "flo.cex")

        # read in the resulting file
        with open(result_path, "r") as df:
            # get alignment result
            data = df.readlines()

        # delete result file
        os.remove(result_path)

        # seperate metadata and data
        meta, data = data[:2], data[2:]

        # parse and format all metadata lines
        res = [[j for j in re.sub(".*\t", "", str(i).strip()).split("|") if j != ''] for i in meta]
        res_dict = {}
        try:
            # for each entry, append tier metadata
            for r in res:
                res_dict[r[2]] = r[4:]
            # and finally, log mmse
            MMSE = int(res_dict[READ[1:]][-1])
            TYPE = res_dict[READ[1:]][1]
        except (ValueError, IndexError, KeyError):
            print(checkfile)
            MMSE=None
            TYPE=None

        # conform result with tab-seperated beginnings
        result = []
        # for each value
        for value in data:
            # if the value continues from the last line
            if value[0] == "\t":
                # pop the result
                try: 
                    res = result.pop()
                except IndexError:
                    print(checkfile)
                # append
                res = res.strip("\n") + " " + value[1:]
                # put back
                result.append(res)
            else:
                # just append typical value
                result.append(value)

        # new the result
        result = [re.sub(r"\x15(\d*)_(\d*)\x15", r"|pause|\1_\2|pause|", i) for i in result] # bullets
        result = [re.sub("\(\.+\)", "", i) for i in result] # pause marks (we remove)
        result = [re.sub("\.", "", i) for i in result] # doduble spaces
        result = [re.sub("  ", " ", i).strip() for i in result] # doduble spaces
        result = [re.sub("\[.*?\]", "", i).strip() for i in result] # doduble spaces
        result = [[j.strip().replace("  ", " ")
                for j in re.sub(r"(.*?)\t(.*)", r"\1??\2", i).split('??')]
                for i in result] # tabs

        # get paired results
        aligned_results = []
        aligned_mor = []
        aligned_gra = []

        # parse various tiers
        main = None
        mor = None
        gra = None
        xwor = None

        # pair up results
        for i in range(0, len(result)):
            # if we have a full result, first append it
            if main and mor and gra and xwor:
                try:
                    aligned_results.append((main[0][:-1], xwor[1]))
                    aligned_mor.append(mor)
                    aligned_gra.append(gra)
                except IndexError:
                    pass

            # get result
            current = result[i]

            # if we have a new tier, parse it like such 
            if current[0][0] == "*":
                main = current.copy()
                mor = None
                gra = None
                xwor = None
            elif current[0] == "%mor:":
                mor = current.copy()[1]
            elif current[0] == "%gra:":
                gra = current.copy()[1]
            elif current[0] == "%xwor:" or current[0] == "%wor:":
                xwor = current.copy()

        # extract pause info
        wordinfo = []
        # record offset
        offset = 0
        lastend = 0

        # calculate final results, which skips any in-between tiers
        # isolate *PAR speaking time only. We do this by tallying
        # a running offset which removes any time between two *PAR
        # tiers, which may include an *INV tier
        for tier, result in aligned_results:
            # set the start as zero
            start = None
            lasttoken = None
            # split tokens
            for token in result.split(" "):
                # if pause, calculate pause
                if token != "" and token[0] == "|":
                    # split pause 
                    res = token.split("_")
                    # get pause values
                    res = [int(re.sub("\+.*", "", i.replace("|pause|>", "").replace("|pause|", "")))
                           for i in token.split("_")]
                    # if to be saved, save
                    if tier == READ:
                        # if not start, set the start
                        if not start:
                            start = res[0]
                            offset += start-lastend # append the differenc
                            # append result
                        wordinfo.append((lasttoken, res[0]-offset, res[1]-offset))
                        # set lastend
                        lastend = res[1]

                # save last token
                lasttoken = token.replace("+","").replace("<","").replace(">","")

        # create wordframe
        wordframe = pd.DataFrame(wordinfo)
        try:
            wordframe.columns=["word", "start", "end"]
        except ValueError:
            continue

        # create a variatino on LubetichSagae to extract dependency parse features
        # parse mor line
        def parse_mor(line):
            # change ~ to space to split combined characters
            line = line.replace("~", " ")
            # split the line by space to tokenize into words
            line = line.split(" ")
            # remove all conjugation annotations
            line = [re.sub("\&.*", "", i) for i in line] 
            # split PoS and words
            line = [i.split("|") for i in line]
            # if we have danging final lines, we just filter
            # ensuring that every element has 2 items
            line = [[i[0], i[1]] for i in line if len(i) >= 2]
            # seperate parts of speech and text
            # we don't actually care about the text.
            pos, _ = zip(*line)
            # return parts of speech only
            return pos

        # parse gra line
        def parse_gra(line, mor_line):
            # we first seperate all relations by space
            line = line.split(' ')
            # and we want to extract relationships and codes
            # remove root for lookup table as well, because
            # other things point to root but root points to
            # nothing
            #
            # We also cut out the punctuation mark.
            line = [[(int(i.split('|')[0]),
                    int(i.split('|')[1])),
                    i.split('|')[2]] for i in line
                    if (i.split("|")[-1] != "ROOT") and
                    (i.split("|")[-1] !="PUNCT")]

            # extract sentence parse tree information
            parse = (i[1] for i in line)

            try:
                # extract parts of speech informiatn
                pos = [mor_line[i[0][0]-1] for i in line]

                # replace lookup information with parts of speech
                # parsed from the mor line
                line = [(mor_line[i[0][0]-1],
                         mor_line[i[0][1]-1],
                         i[1]) for i in line]
            except IndexError:
                # replace lookup information with parts of speech
                # parsed from the mor line
                # NOF is a token that's not found
                line = []
                pos = [] 

            # return the prepared line
            return line, parse, pos

        # parse the entire database to generate the equivalent of
        # feature #4 by LubetichSagae which combines #s 2,3,4
        # together
        # extracted_syntactic_features = []
        extracted_relational_features = []
        extracted_pos_features = []

        for mor, gra in zip(aligned_mor, aligned_gra):
                        # calculate mor line
            mor_line = parse_mor(mor)
                        # calculate feature
            _, relation, pos  = parse_gra(gra, mor_line)

            # append to list
            # extracted_syntactic_features.append(syntax)
            extracted_relational_features.append(relation)
            extracted_pos_features.append(pos)

        # we encode everything the global defaultdict, which will add to the dictionary if it
        # doesn't exist already
        # encoded_syntax_features = [[tokens[j] for j in i] for i in extracted_syntactic_features]
        encoded_relational_features = [[tokens[j] for j in i] for i in extracted_relational_features]
        encoded_pos_features = [[tokens[j] for j in i] for i in extracted_pos_features]

        # save the syntax features
        with open(repath_file(checkfile, OUTDIR).replace(".cha", "-meta.bin"), "wb") as df:
                        # dump the syntax features
            if MMSE:
                pickle.dump({"relational": encoded_relational_features,
                            "pos": encoded_pos_features,
                            "mmse": MMSE}, df)
            else:
                pickle.dump({"relational": encoded_relational_features,
                            "pos": encoded_pos_features}, df)

        # write the final output file
        wordframe.to_csv(repath_file(checkfile, OUTDIR).replace(".cha", "-wordframe.csv"))

do(DATADIR_AD, OUTDIR_AD)
do(DATADIR_C, OUTDIR_C)

# dump the syntax token lookup table
with open(TOKENDIR, "wb") as df:
    # dump the frozen syntax features
    # we reverse the dictionary as the typical usage
    pickle.dump({"tokens": {tokens[k] : k for k in tokens}}, df)


