# os utilities
import os
from posixpath import dirname

# random for shuffle!
import random

# import pathlib
from pathlib import Path

# import globbing tools
import glob
from numpy.lib.stride_tricks import sliding_window_view

# import pandas
import pandas as pd # type: ignore

#################################################

# set the path for data, this changes based on
# the experiement we are running
DATA_PATH =  "./data/transcripts_pauses/alignedpitt-7-11/" # in path
OUT_PATH = "./data/transcripts_pauses/alignedpitt-7-11-flucalc.bat" # out path
WINDOWED_PATH = "./data/transcripts_pauses/alignedpitt-7-11-flucalc-windowed.bat" # out path

DEMENTIA_META = "./data/transcripts_pauses/alignedpitt-7-8-flucalc/dementia.xlsx"
CONTROL_META = "./data/transcripts_pauses/alignedpitt-7-8-flucalc/control.xlsx"

WINDOW_SIZE =  5
TESTING_SPLIT = 5  # testing split (patients kper class)

#################################################

# glob for the control files and dementia files
control_flo_outs = glob.glob(os.path.join(DATA_PATH, "control/*.txt"))
dementia_flo_outs = glob.glob(os.path.join(DATA_PATH, "dementia/*.txt"))

#################################################

# read the files as a csv and clean

def read_and_clean(files, target):
    """read a list of .txt utterances and clean them into a dataframe

    Arguments:
        files (list[string]): list of files to clean
        target (any): the thing to go into the target column

    Returns:
        a DataFrame with columns trial utterance and target
        specificed in the "target" argument

    """

    # create an array for dfs
    dfs = []

    # for each control file
    for data_file in files:
        # read the datafile as a single column csv
        try: 
            data = pd.read_csv(data_file, sep="`", header=None)
        except:
            print(f"Read error on {data_file}, skipping.")
            continue
        
        # label column as utterance
        data.columns = ["utterance"]
        # add another column representing file name
        data["trial"]=Path(data_file).stem
        # and another representing target
        data["target"] = target
        # let's reorder the columns a little
        data = data[["trial", "utterance", "target"]]
        # append!
        dfs.append(data)

    # finally, assemble control data
    result = pd.concat(dfs)

    # lastly, clean the utterance by strip
    result["utterance"] = result["utterance"].apply(lambda x:x.strip())

    return result

# metadata
meta_control = pd.read_excel(CONTROL_META, index_col=0)
meta_control.dropna(axis=1, inplace=True)
meta_control = meta_control.iloc(axis=1)[11:]

meta_dementia = pd.read_excel(DEMENTIA_META, index_col=0)
meta_dementia.dropna(axis=1, inplace=True)
meta_dementia = meta_dementia.iloc(axis=1)[11:]

# normalize all mor incidies
# control
mor_control = meta_control[["mor_Utts",
                            "mor_Words",
                            "mor_syllables"]].apply(lambda x: (x-x.mean())/x.std(),
                                                    axis=0)
meta_control[["mor_Utts",
              "mor_Words",
              "mor_syllables"]] = mor_control
# dementia
mor_dementia = meta_dementia[["mor_Utts",
                              "mor_Words",
                              "mor_syllables"]].apply(lambda x: (x-x.mean())/x.std(),
                                                      axis=0)
meta_dementia[["mor_Utts",
               "mor_Words",
               "mor_syllables"]] = mor_dementia

# note that, for the "target" colmumn, 1 is Dementia and 0 is control

## control files ##
control = read_and_clean(control_flo_outs, 0)
control.reset_index(drop=True, inplace=True)

## dementia files ##
dementia = read_and_clean(dementia_flo_outs, 1)
dementia.reset_index(drop=True, inplace=True)

# get metadata for control
meta_control_table = meta_control.loc[control["trial"].apply(lambda x:x+".cha")]
meta_dementia_table = meta_dementia.loc[dementia["trial"].apply(lambda x:x+".cha")]

# reset indicies
meta_control_table.reset_index(drop=True, inplace=True)
meta_dementia_table.reset_index(drop=True, inplace=True)

# concatenate tables
control = pd.concat([control, meta_control_table], axis=1)
dementia = pd.concat([dementia, meta_dementia_table], axis=1)

# concat!
data = pd.concat([control, dementia])
# reset index
data = data.reset_index(drop=True)

#################################################

# finally, make `trial` a part of the multiindex
# set the index to the multiindex
data_indicies = zip(data["trial"], data.index)
data.index = pd.MultiIndex.from_tuples(data_indicies, names=["trial", "sample"])
# drop the existing trial column
data = data.drop(columns=["trial"])

#################################################

# get indicies
control_indicies = list(set([i[0] for i in data[data["target"] == 0].index]))
dementia_indicies = list(set([i[0] for i in data[data["target"] == 1].index]))

# shuffle
randomness = random.Random(0)
randomness.shuffle(control_indicies)
randomness.shuffle(dementia_indicies)

# crop out the indicies to ensure length
desired_length = min(len(control_indicies), len(dementia_indicies))
control_indicies = control_indicies[:desired_length]
dementia_indicies = dementia_indicies[:desired_length]

# parcel out the last TESTING_SPLIT from each for testing
training_control_indicies = control_indicies[:-TESTING_SPLIT]
training_dementia_indicies = dementia_indicies[:-TESTING_SPLIT]

# get testing indicies
testing_indicies = control_indicies[-TESTING_SPLIT:]+dementia_indicies[-TESTING_SPLIT:]

# parcel out training data
training_control_data = data.loc[training_control_indicies]
training_dementia_data = data.loc[training_dementia_indicies]

# cut the htraining data down
desired_training_length = min(len(training_control_data), len(training_dementia_data))
training_data = pd.concat([training_control_data.iloc[:desired_training_length],
                           training_dementia_data.iloc[:desired_training_length]])

# parcel out testing data
testing_data = data.loc[testing_indicies]

# add labels
training_data["split"] = "train"
testing_data["split"] = "test"

# concatenate
data = pd.concat([training_data, testing_data])

# final shuffle
data_shuffled = data.iloc[randomness.sample(range(len(data)), len(data))]

#################################################

# concatenated results
results = []
trials = []

# # create windowed results
for trial, frame in data.groupby(level=0):
    if WINDOW_SIZE:
        ws = WINDOW_SIZE
    else:
        ws = len(frame)-1
    # for each slice index
    for i in range(0, len(frame)-ws, 1):
        # for each slide of the data
        slice = frame.iloc[i:i+ws]
        # get the concatenated string
        utterance_concat = " ".join(slice["utterance"])

        # if length is zero, then ignore
        if len(slice) == 0:
            continue

        # create final metadata slice
        final_slice = slice.drop(columns=["utterance"]).iloc[0]
        final_slice["utterance"] = utterance_concat

        # append the results
        results.append(final_slice)
        # tell results about trials
        trials.append(trial)


# TODO
# create windowed results
data_windowed = pd.DataFrame(results)
# create new window index
data_windowed_tuples = [(i[1], i[0]) for i in list(enumerate(trials))]
data_windowed_index = pd.MultiIndex.from_tuples(data_windowed_tuples,
                                                names=["trial", "sample"])
data_windowed.index = data_windowed_index

# final shuffle
data_windowed_shuffled = data_windowed.iloc[randomness.sample(range(len(data_windowed)), len(data_windowed))]

# dump to data 
data_shuffled.to_pickle(OUT_PATH)
data_windowed_shuffled.to_pickle(WINDOWED_PATH)


