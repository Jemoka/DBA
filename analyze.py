# glob
import glob

# pathing
import os
from pathlib import Path

# pandas
import pandas as pd
from pandas.core.indexers.utils import length_of_indexer
from transformers.utils.dummy_pt_objects import DPTForSemanticSegmentation

# random tools from sklearn
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# stats
from scipy.stats import kstest, pearsonr

# plotting
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt

# get all dirs
CONTROL_DIR = "/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-12/control/"
DEMENTIA_DIR = "/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-12/dementia/"
OUT_DIR = "/Users/houliu/Documents/Projects/DBA/data/wordinfo/pitt-07-12.csv"

# get val test split
TEST_SPLIT = 0.1

# get all files
control_files = glob.glob(os.path.join(CONTROL_DIR, "*.csv"))
dementia_files = glob.glob(os.path.join(DEMENTIA_DIR, "*.csv"))

# collect targets
# for each file
def process_targets(files):
    result = []

    # for each file
    for f in files:
        # read the csv
        df = pd.read_csv(f, index_col=0)
        # name columns
        df.columns = ["start", "end"]
        # change units to seconds
        df = df/1000

        # get pauses
        diffs = (df["start"].shift(-1)-df["end"])
        diffs.iloc[-1] = 0
        pauses = diffs[diffs!=0]

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
        speech_duration = (df.end-df.start).sum()

        if silence_duration > 0:
            voice_silence_ratio = speech_duration/silence_duration
        else:
            voice_silence_ratio = 0 # this is almost max

        # # pause metadata (not mentinoed)
        # inter_pause_distance = df.diff().iloc[1:]["start"]
        # mean_inter_pause_distance = inter_pause_distance.mean()
        # max_inter_pause_distance = inter_pause_distance.max()
        # inter_pause_distance_std = inter_pause_distance.std()

        # create metadata column
        data = pd.Series({
            "max_pause": max_pause,
            "mean_pause": mean_pause,
            "pause_std": pause_std,
            "duration": duration,
            "verbal_rate": verbal_rate,
            "silence_duration": silence_duration,
            "speech_duration": speech_duration,
            "voice_silence_ratio": voice_silence_ratio,
        })

        # append data
        result.append(data)

    # return result
    return result

# process control
control = process_targets(control_files)
control = pd.DataFrame(control)
control = control.dropna()
control["target"] = 0

# process dementia
dementia = process_targets(dementia_files)
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
# shuffle again
data = data.sample(frac=1)




#### Statisics and Simple Analysis ####

# data
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

wang_stat_results

# analyze pause results
PAUSE = ["max_pause", "mean_pause", "pause_std", "verbal_rate"] 
pause_results_ad, pause_results_control = describe_variables(data, PAUSE)
pause_stat_results = analyze_variables(data, PAUSE)

pause_stat_results

#### ML and Classification ####

# train data
train_data = data.iloc[:-int(TEST_SPLIT*len(data))]
test_data = data.iloc[-int(TEST_SPLIT*len(data)):]

# in and out data
in_data = train_data.iloc(axis=1)[:-1]
out_data = train_data.iloc(axis=1)[-1]

in_test = test_data.iloc(axis=1)[:-1]
out_test = test_data.iloc(axis=1)[-1]

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

# random classifier test
clsf = SVC()
clsf.fit(in_data, out_data)
clsf.score(in_test, out_test)

# feature selection with ANOVA
k_best = SelectKBest(f_classif, k=2)
k_best.fit(in_data, out_data)
best_features = k_best.get_feature_names_out()

# random forest
clsf = RandomForestClassifier(criterion="entropy")
clsf = clsf.fit(in_data, out_data)
clsf.score(in_test, out_test)

#### trash ####

# # run PCA
# pca = PCA(n_components=2)
# in_pca = pca.fit(in_data)
# data_pca = pca.transform(in_concat.iloc(axis=1)[:-1])

# # plot PCA
# pca_plot = sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=out_concat, style=in_concat.split)
# pca_plot.set(xlabel="PCA1", ylabel="PCA2")
# plt.show()

