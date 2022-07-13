# os utilities
import os
# pathing
import pathlib
# regex
import re
# glob
import glob

import pandas as pd

from pandas.core.generic import _align_as_utc

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob.glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, pathlib.Path(file_path).name)

# path of CLAN
CLAN_PATH=""

# file to check
DATADIR="/Users/houliu/Documents/Projects/DBA/data/raw/pitt-07-12/dementia/"
OUTDIR="/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-12/dementia"

# get output
files = globase(DATADIR, "*.cha")

# for each file
for checkfile in files:

    # run flo on the file
    CMD = f"{os.path.join(CLAN_PATH, 'flo +t%xwor -t*INV')} {checkfile} >/dev/null 2>&1"
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

    # conform result with tab-seperated beginnings
    result = []
    # for each value
    for value in data:
        # if the value continues from the last line
        if value[0] == "\t":
            # pop the result
            res = result.pop()
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
    result = [re.sub(".*?\\t", "", i) for i in result] # tabs
    result = [re.sub("\.", "", i) for i in result] # doduble spaces
    result = [re.sub("  ", " ", i).strip() for i in result] # doduble spaces
    result = [re.sub("\[.*?\]", "", i).strip() for i in result] # doduble spaces

    # get paired results
    aligned_results = []

    # pair up results
    for i in range(0, len(result)-3, 3):
        # append paired result
        aligned_results.append(result[i+2])

    # extract pause info
    wordinfo = []
    for result in aligned_results:
        # collect result token
        start = None
        end = None

        # split tokens
        for token in result.split(" "):
            # if pause, calculate pause
            if token != "" and token[0] == "|":
                # split pause 
                res = token.split("_")
                # get pause values
                res = [int(i.replace("|pause|>", "").replace("|pause|", "")) for i in token.split("_")]
                wordinfo.append((res[0], res[1]))

    wordframe = pd.DataFrame(wordinfo)
    try:
        wordframe.columns=["start", "end"]
    except ValueError:
        continue

    # write the final output file
    # wordframe.to_csv(repath_file(checkfile, OUTDIR).replace("cha", "csv"))

