# glob
from contextlib import ContextDecorator
import glob

# pathing
import os
from pathlib import Path

# pandas and numpy
import pandas as pd
import numpy as np
from pandas._libs.tslibs.period import DIFFERENT_FREQ
from pandas.core.indexers.utils import length_of_indexer
from transformers.utils.dummy_pt_objects import DPTForSemanticSegmentation

# random tools from sklearn
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier

# stats
from scipy.stats import kstest, pearsonr

# import unpickling
import pickle

# plotting
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt

# get all dirs
CONTROL_DIR = "/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-14/control/"
DEMENTIA_DIR = "/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-14/dementia/"
# OUT_DIR = "/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-13.csv"
OUT_DIR = None

# get val test split
TEST_SPLIT = 0.1

# get all files
control_files = glob.glob(os.path.join(CONTROL_DIR, "*.csv"))
control_syntax = glob.glob(os.path.join(CONTROL_DIR, "*.bin"))
control_lookup = os.path.join(CONTROL_DIR, "tokens.bin")

dementia_files = glob.glob(os.path.join(DEMENTIA_DIR, "*.csv"))
dementia_syntax = glob.glob(os.path.join(DEMENTIA_DIR, "*.bin"))
dementia_lookup = os.path.join(DEMENTIA_DIR, "tokens.bin")

# collect targets
# for each file
def process_targets(files, syntax):
    # verbal rate interpolation rate for trend
    VERBAL_SHIFT = 50
    PAUSE_SHIFT = 1

    result = []

    # for each file
    for f,s in zip(sorted(files), sorted(syntax)):
        # read the csv
        df = pd.read_csv(f, index_col=0)
        # name columns
        df.columns = ["word", "start", "end"]
        # change units to seconds
        df[["start", "end"]] = df[["start", "end"]]/1000

        # get pauses
        diffs = pd.DataFrame({"start":df["end"], "end":df["start"].shift(-1)})
        diffs["pauses"] = diffs.end-diffs.start

        diffs = diffs.dropna()
        pauses = diffs[diffs.pauses!=0].pauses

        # pause data -- Antonsson 2021
        max_pause = pauses.max()
        mean_pause = pauses.mean()
        pause_std = pauses.std()
        silences_per_words = len(pauses)/len(df)

        # speaking duration -- Balagopalan 2021
        duration = (df.iloc[-1].end - df.iloc[0].start)

        # verbal rate data -- Wang 2019, Lindsay 2021
        verbal_rate = (len(df) / duration)
        silence_duration = pauses.sum()

        # vocabulary data -- Wang 2019
        total_words = len(set(df["word"]))

        # speech rate data -- Beltrami 2014
        phonation_time = total_words/duration

        if silence_duration > 0:
            voice_silence_ratio = (df.end-df.start).sum()/silence_duration
        else:
            voice_silence_ratio = 0 # this is almost max

        # verbal rate trend calculation -- author's invention
        rate_interpolated = ((VERBAL_SHIFT+1)/(df["end"].shift(-VERBAL_SHIFT)-df["start"])).dropna()
        if len(rate_interpolated) > 0:
            try: 
                fit = np.polyfit(x=range(len(rate_interpolated)), y=rate_interpolated, deg=1)
            except:
                fit = [0,0]
        else:
            fit = [0,0]

        # pause rate calculation -- Beltrami 2014
        # also called "voice to silence ratio"
        try:
            pause_rate = total_words/len(pauses)
        except ZeroDivisionError:
            pause_rate = total_words

        # decode the syntax output
        with open(s, 'rb') as df:
            syntax_parsed = pickle.load(df)

        # flatten output
        syntax_parsed_flattened = np.array(sum(syntax_parsed, []))
        s,x = syntax_parsed_flattened.shape

        # create metadata column
        data = pd.Series({
            "max_pause": max_pause,
            "mean_pause": mean_pause,
            "pause_std": pause_std,
            "total_words": total_words,
            "verbal_rate": verbal_rate,
            "verbal_rate_trend": fit[0],
            "verbal_rate_interpolated": rate_interpolated,
            "phonation_time": phonation_time,
            "silence_duration": silence_duration,
            "speech_duration": duration,
            "voice_silence_ratio": pause_rate,
            "syntax": syntax_parsed_flattened.reshape(s*x)
        })

        # append data
        result.append(data)

    # return result
    return result

# process control
control = process_targets(control_files, control_syntax)
control = pd.DataFrame(control)
control = control.dropna()
control["target"] = 0

# process dementia
dementia = process_targets(dementia_files, dementia_syntax)
dementia = pd.DataFrame(dementia)
dementia = dementia.dropna()
dementia["target"] = 1

# shuffle and crop to same
control = control.sample(frac=1)
dementia = dementia.sample(frac=1)
control = control[:min(len(control), len(dementia))]
dementia = dementia[:min(len(control), len(dementia))]

# concat
data = pd.concat([dementia, control])
data.reset_index(drop=True, inplace=True)
# pad the syntax to max utterance length
max_utterance_length = data["syntax"].apply(lambda x : len(x)).max()

# utility to pad sequence
def pad_seq(x):
    # create the pad arr
    pad_arr = np.array([-1 for _ in range(max_utterance_length-len(x))])
    # if we need to pad, pad
    if (max_utterance_length-len(x)) > 0:
        res = np.concatenate((x, pad_arr))
    # else, do nothing
    else:
        res = x
    return res

# set the padded result back
data["syntax_padded"] = data["syntax"].apply(pad_seq)

# shuffle again
data = data.sample(frac=1)

#### Statisics and Simple Analysis ####

# data
if OUT_DIR:
    data.to_csv(OUT_DIR, index=False)

# analyzer tools
# mean and std
def describe_variables(data, variables):
    # get AD results and calculate
    results_ad = data[data.target==1][variables].apply(lambda x:(x.mean(), x.std()), axis=0)
    # reset index
    results_ad.index = ["mean", "std"]
    # transpose
    results_ad = results_ad.transpose()

    # get control results and calculate
    results_control = data[data.target==0][variables].apply(lambda x:(x.mean(), x.std()), axis=0)
    # reset index
    results_control.index = ["mean", "std"]
    # transponse
    results_control = results_control.transpose()

    return results_ad, results_control

# statistical difference
def analyze_variables(data, variables):
    # get AD results and calculate
    results_ad = data[data.target==1][variables]
    # get control results and calculate
    results_control = data[data.target==0][variables]

    # collect results
    ks = {}
    pc = {}

    # for each variable, perform ks test
    for variable in variables:
        # get variable
        ad = results_ad[variable]
        control = results_control[variable]
        # perform test
        _, p = kstest(ad, control)
        # dump
        ks[variable] = p

    # for each variable, coorelate with results
    for variable in variables:
        # get variable
        ad = results_ad[variable]
        control = results_control[variable]
        # create targets
        targets = [1 for _ in range(len(ad))] + [0 for _ in range(len(control))]
        # perform test
        _, p = pearsonr(pd.concat([ad, control]), targets)
        # dump
        pc[variable] = p

    return {"kstest": ks, "pearson": pc}

# analyze wang results
WANG = ["silence_duration", "speech_duration", "voice_silence_ratio", "verbal_rate"] 
wang_results_ad, wang_results_control = describe_variables(data, WANG)
wang_stat_results = analyze_variables(data, WANG)

# analyze pause results
PAUSE = ["max_pause", "mean_pause", "pause_std", "verbal_rate"] 
pause_results_ad, pause_results_control = describe_variables(data, PAUSE)
pause_stat_results = analyze_variables(data, PAUSE)

#### ML and Classification ####

# train data
train_data = data.iloc[:-int(TEST_SPLIT*len(data))]
test_data = data.iloc[-int(TEST_SPLIT*len(data)):]

# in and out data
in_data = train_data.drop(columns=["target"])
out_data = train_data["target"]

in_test = test_data.drop(columns=["target"])
out_test = test_data["target"]

# concatenate data
# in data
# copy frames
in_copy = in_data.copy()
test_in_copy = in_test.copy()

# set split
in_copy["split"] = "train"
test_in_copy["split"] = "test"

# put together
in_concat = pd.concat([in_copy, test_in_copy])

# out data
out_concat = pd.concat([out_data, out_test])

# create 3d syntax input array, which we will flatten
in_data_syntax = np.array(in_data["syntax_padded"].to_list())
in_test_syntax = np.array(in_test["syntax_padded"].to_list())

# random forest
clsf = RandomForestClassifier()
clsf = clsf.fit(in_data_syntax, out_data)
clsf.score(in_test_syntax, out_test)

# decision tree
clsf = DecisionTreeClassifier()
clsf = clsf.fit(in_data_syntax, out_data)
clsf.score(in_test_syntax, out_test)

plot_tree(clsf)
plt.show()

# random classifier test
clsf = SVC(kernel='poly')
clsf = clsf.fit(in_data_syntax, out_data)
clsf.score(in_test_syntax, out_test)

# KNN
clsf = KNeighborsClassifier(2)
clsf = clsf.fit(in_data_syntax, out_data)
clsf.score(in_test_syntax, out_test)


# plot
feature_plot = sns.scatterplot(data=in_concat, x="verbal_rate", y="silence_duration", hue=out_concat, style="split")
plt.show()



#### PCA ####

# collect pca data
# norm_data = data.drop(columns=["verbal_rate_interpolated", "pause_rate_interpolated"])
# norm_data.iloc(axis=1)[:-1] = norm_data.iloc(axis=1)[:-1].apply(lambda x:(x-x.mean())/x.std(), axis=0)

# run PCA
pca = PCA(n_components=2)
in_pca = pca.fit_transform(in_data_syntax)
# data_pca = pca.transform(norm_data)

# plot PCA
pca_plot = sns.scatterplot(x=in_pca[:,0], y=in_pca[:,1], hue=out_data)
pca_plot.set(xlabel="PCA1", ylabel="PCA2")
plt.show()





