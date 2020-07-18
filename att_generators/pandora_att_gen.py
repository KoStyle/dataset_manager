import argparse
import math
import os
import pickle
import numpy as np
import pandas as pd

from att_generators.void_attr_gen import VoidAttGen
from scipy.sparse import csr_matrix, hstack as sparse_hstack


class PandoraAttGen(VoidAttGen):
    datadic={}
    dfdic={}
    feat_names = None



    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser(description='')

        # parser.add_argument("-size", "--size", help="size of the dataset to work on (100,500,700,1000,1500,2000,2500)", type=int)
        parser.add_argument("-data_path", "--data_path",
                            help="Path to the \"data\" folder of this distribution (contains the unfiltered comments, user metadata - author_profiles, fold splits, and some precomputed enneagram/mbti features for the big5 regressions.")
        parser.add_argument("-repeat", "--repeat",
                            help="repeat to work on (0 - 4) or -1 for all repeats (default 0). Each repeat represents a different random split for the five folds. To get results from the paper use repeat 0.",
                            type=int, default=0)
        parser.add_argument("-label", "--label",
                            help="which column to predict, (any column but most often -- introverted, intuitive, thinking, perceiving, agreeableness, openness, conscientiousness, extraversion, neuroticism, enneagram_type, age, is_female. Additional acceptable values are 'allmbti', 'allbig5', these run the models for multiple columns all at once.",
                            type=str)
        parser.add_argument("-tasktype", "--tasktype",
                            help="can be 'classification' or 'regression' (default classification)", type=str)
        parser.add_argument("-folds", "--folds",
                            help="which set of folds to use, you probably want 'mbti', 'big5_scores', 'big5_percentiles', 'enneagram', 'age' or 'gender', but any set of folds will work as long as it is compatible with your chosen labels (e.g., don't use 'mbti' folds for predicting a big5 style column, if you do that, something can and WILL go terribly horribly wrong), this parameter also does some prefiltering (see --filters for the nifty details)",
                            type=str)
        parser.add_argument("-feats", "--feats",
                            help="comma separated list of feature types used, can be 1gram, 12gram, 123gram, charngram, style, liwc, age, gender, mbtipred, big5pred (use only one 'gram' style entry if you use more only the first one is considered) ",
                            type=str)
        parser.add_argument("-variant", "--variant",
                            help="string describing a variant of the experiment, such as FemaleOnly or Over21RegressionPercentilesOnly",
                            type=str)
        parser.add_argument("-model", "--model",
                            help="optional, which model to use, 'lr' (logistic/linear regression), or 'et' (extratrees classifier/regressor) or 'dummy' (most frequent class for classification or train set mean for regression), default is lr",
                            default="lr", type=str)
        parser.add_argument("-specificfold", "--specificfold",
                            help="optional, speicifies which particular fold to run (single number 0 to 4), or -1 to run all folds, default is -1",
                            default=-1, type=int)
        parser.add_argument("-topfeats", "--topfeats",
                            help="optional, number of top features from feature selection to dump to file, -1 to dump all used feats, default is -1",
                            default=-1, type=int)
        return parser.parse_args(args)

    @staticmethod
    def init_values_and_stuff_mtbi():
        argstr = ['-data_path', "C:\\Users\\konom\\Downloads\\pandora_baseline\\data", '-label', 'allmbti', '-tasktype', 'classification', '-folds', 'mbti', '-feats', '1gram', '-model', 'lr', '-variant', 'LR-N']
        args = PandoraAttGen.parse_args(argstr)
        unm = pickle.load(open(os.path.join(args.data_path, "unames.pickle"), "rb"))
        txt = []
        labels = ["introverted", "intuitive", "thinking", "perceiving"]
        for label_name in labels:
            PandoraAttGen.datadic[label_name] = PandoraAttGen.load_data(unm, txt, os.path.join(args.data_path, "author_profiles.csv"), label_name, args.tasktype, args.folds, 0, args)
            PandoraAttGen.dfdic[label_name], PandoraAttGen.feat_names, extra_feats, extra_feat_names = PandoraAttGen.precompute_or_load_feats(PandoraAttGen.datadic[label_name], args.data_path, args)
        return None

    @staticmethod
    def load_data(unames, texts, labels_path, label_name, label_type, fold_grp, repeat, args):  # percentiles or scores THAT PARAMETER IS DEPRECATED DOES NOTHING NOW
        print("Loading metadata ...")
        # load labels
        ldf = pd.read_csv(labels_path)

        auth2label = dict(zip(list(ldf["author"]), list(ldf[label_name])))
        labels = [auth2label[x] for x in unames]

        # generate folds (or load from disk where available)
        folds_filename = os.path.join(args.data_path, "folds.csv")

        print("Loading the folds file ...")
        if os.path.isfile(folds_filename):
            fdf = pd.read_csv(folds_filename)

            fdf = fdf[fdf["group"] == fold_grp]
            fdf = fdf[fdf["trait"] == label_name]
            fdf = fdf[fdf["repeat"] == repeat]

            auth2fold = dict(zip(list(fdf["author"]), list(fdf["fold"])))
            folds = []
            nonfails, fails = 0, 0
            for x in unames:
                if x in auth2fold:
                    nonfails += 1
                    folds.append(auth2fold[x])
                else:
                    fails += 1
                    folds.append(None)
            assert len(folds) == len(unames)  # this also has to be true
            assert len(folds) == len(labels)
            if len(
                    auth2fold) != nonfails:  # if this is true then there are no examples that exist in the folds list but not in the data
                print(
                    "Warning, there are some users in the folds file that are not in the data (maybe you removed mbti subreddits from the data but not from the folds), these users will be ignored!")

            for i in range(len(folds)):
                if folds[i] is None:
                    labels[
                        i] = math.nan  # later, everything where the label is None is thrown out, including these users for which we have no fold indexes
        else:
            raise Exception("Fold file not found.")
        ret_df = pd.DataFrame(list(zip(unames, folds, labels)), columns=['author', 'fold', 'label'])
        return (ret_df)


    @staticmethod
    def precompute_or_load_feats(data, feats_id_prefix, args):
        # *** all *** means all except n-gram feats
        all_feats_names = []
        all_feats_list = []

        text_feats_names = []
        text_feats_matrix = None
        grams_finished = False
        slen_finished = False
        argssplitlist = [x.strip() for x in args.feats.split(",")]
        PATH_PREDICTION_FEATS = os.path.join(args.data_path, "mbti_enne_pred.csv")

        for feats_type in argssplitlist:
            if "gram" in feats_type and not grams_finished:

                feats_filename = os.path.join(feats_id_prefix, "feats.pickle")
                vocab_filename = os.path.join(feats_id_prefix, "vocab.pickle")
                try:
                    print("Loading precomputed features from disk ...")
                    gram_feats = pickle.load(open(feats_filename, "rb"))
                    gram_feat_names = pickle.load(open(vocab_filename, "rb"))
                except:
                    raise Exception("Something went wrong when loading the n-gram features files ...")

                text_feats_names = gram_feat_names
                text_feats_matrix = gram_feats
                grams_finished = True

            if feats_type == "mbtipred":
                mbti_df = pd.read_csv(PATH_PREDICTION_FEATS)
                joined = data[["author"]].merge(mbti_df, on="author", how="left")
                mbti_feat_names = ["introverted_pred", "intuitive_pred", "thinking_pred", "perceiving_pred"]
                mbti_feats = csr_matrix(np.array(joined[mbti_feat_names]))
                all_feats_names += mbti_feat_names
                all_feats_list.append(mbti_feats)

            if feats_type == "ennepred":
                enne_df = pd.read_csv(PATH_PREDICTION_FEATS)
                joined = data[["author"]].merge(enne_df, on="author", how="left")
                enne_feat_names = ["pred_e_type_" + str(n) for n in range(1, 10)]
                enne_feats = csr_matrix(np.array(joined[enne_feat_names]))
                all_feats_names += enne_feat_names
                all_feats_list.append(enne_feats)

        all_feats_matrix = csr_matrix(sparse_hstack(all_feats_list)) if len(all_feats_names) > 0 else None
        if all_feats_matrix is not None:
            assert all_feats_matrix.shape[1] == len(all_feats_names)
            print(all_feats_matrix.shape)
        if text_feats_matrix is not None:
            assert text_feats_matrix.shape[1] == len(text_feats_names)
            print(text_feats_matrix.shape)
        return (text_feats_matrix, text_feats_names, all_feats_matrix, all_feats_names)
