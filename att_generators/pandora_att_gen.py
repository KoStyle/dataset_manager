import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler

from att_generators.void_attr_gen import VoidAttGen
from constants import TYPE_LST, TYPE_NUM, PANDORA_DICT, PANDORA_CORPUS, PANDORA_MODELS


class PandoraSetUp():
    def __init__(self, model, fs, tfidfer, data=None, data_feats=None):
        self.model = model
        self.fs: SelectKBest = fs
        self.tfidfer: TfidfTransformer = tfidfer
        self.data = data
        self.data_feats = data_feats


# TODO Maybe clean a reestructure this methods a bit maybe (perhaps)
# TODO override __init__ to put the model initialization THEREGODDAMNIT
class PandoraAttGen(VoidAttGen):
    model_dic = {}
    feat_names = None
    label_setups = {}

    @staticmethod
    def gen_pandora_mbti():
        if PandoraAttGen.feat_names is None:
            PandoraAttGen.init_values_and_models_and_stuff()

        def get_pandora_mbti(text):
            labels = ["introverted", "intuitive", "thinking", "perceiving"]

            # Transformation of the matrix according the feat_selection and tfidf normalizer of each model and store the values returned in a list
            att_lst = PandoraAttGen.__get_label_predictions(labels, text)
            return att_lst

        return PandoraAttGen('PANDORA_MBTI', TYPE_LST, get_pandora_mbti)

    @staticmethod
    def gen_pandora_age():
        if PandoraAttGen.feat_names is None:
            PandoraAttGen.init_values_and_models_and_stuff()

        def get_pandora_age(text):
            labels = ["age"]

            # Transformation of the matrix according the feat_selection and tfidf normalizer of each model and store the values returned in a list
            att_lst = PandoraAttGen.__get_label_predictions(labels, text)
            return att_lst[0]  # This generator returns a single value, the age prediction

        return PandoraAttGen('PANDORA_AGE', TYPE_NUM, get_pandora_age)

    @staticmethod
    def gen_pandora_gender():
        if PandoraAttGen.feat_names is None:
            PandoraAttGen.init_values_and_models_and_stuff()

        def get_pandora_gender(text):
            labels = ["is_female"]

            # Transformation of the matrix according the feat_selection and tfidf normalizer of each model and store the values returned in a list
            att_lst = PandoraAttGen.__get_label_predictions(labels, text)
            return att_lst[0]  # This generator returns a single value, the age prediction

        return PandoraAttGen('PANDORA_GENDER', TYPE_NUM, get_pandora_gender)

    @staticmethod
    def __get_label_predictions(label_list, text):
        the_matrix = PandoraAttGen.__text_to_matrix(text)

        # Transformation of the matrix according the feat_selection and tfidf normalizer of each model and store the values returned in a list
        att_lst = []
        for label_name in label_list:
            label_setup: PandoraSetUp = PandoraAttGen.model_dic[label_name]
            matrix_reloaded = label_setup.tfidfer.fit_transform(the_matrix)
            matrix_revolutions = csr_matrix(label_setup.fs.transform(matrix_reloaded))
            att_lst.append(label_setup.model.predict(matrix_revolutions)[0])  # we take the first (and only) value predicted
        return att_lst

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
    def __text_to_matrix(text):
        # Split by words and create a dict with the word frequency
        words = text.split()
        ui_dict = {}
        for word in words:
            if word in ui_dict:
                ui_dict[word] += 1
            else:
                ui_dict[word] = 1

        # Mapping the instance's word frequency (WF) to the words (feats) tracked by Pandora in a list
        vect = []
        for word in PandoraAttGen.feat_names:
            if word in ui_dict:
                vect.append(ui_dict[word])
            else:
                vect.append(0)

        # Use of the list to create a sparse matrix readable by the Pandora models
        rows = []
        cols = []
        mxdata = []
        for i in range(len(vect)):
            if vect[i] != 0:
                rows.append(0)
                cols.append(i)
                mxdata.append(vect[i])
        enter_the_matrix = csr_matrix((mxdata, (rows, cols)), shape=(1, len(PandoraAttGen.feat_names)))
        return enter_the_matrix

    @staticmethod
    def init_values_and_models_and_stuff():
        if os.path.exists(PANDORA_DICT):
            mega_dict = pickle.load(open(PANDORA_DICT, "rb"))
            PandoraAttGen.model_dic = mega_dict[PANDORA_MODELS]
            PandoraAttGen.feat_names = mega_dict[PANDORA_CORPUS]
        else:
            PandoraAttGen.__init_mbti_models()
            PandoraAttGen.__init_gender_model()
            PandoraAttGen.__init_age_model()
            mega_dict = {PANDORA_MODELS: PandoraAttGen.model_dic, PANDORA_CORPUS: PandoraAttGen.feat_names}
            if not os.path.exists(os.path.dirname(PANDORA_DICT)):
                os.makedirs(os.path.dirname(PANDORA_DICT))
            pickle.dump(mega_dict, open(PANDORA_DICT, "wb"))

    @staticmethod
    def __init_mbti_models():
        argstr = '-data_path C:\\Users\\konom\\Downloads\\pandora_baseline\\data -label allmbti -tasktype classification -folds mbti -feats 1gram -model lr -variant LR-N'
        args = PandoraAttGen.parse_args(argstr.split())
        unm = pickle.load(open(os.path.join(args.data_path, "unames.pickle"), "rb"))
        txt = []
        labels = ["introverted", "intuitive", "thinking", "perceiving"]
        for label_name in labels:
            data = PandoraAttGen.load_data(unm, txt, os.path.join(args.data_path, "author_profiles.csv"), label_name, args.tasktype, args.folds, 0, args)
            df, feat_names, extra_feats, extra_feat_names = PandoraAttGen.precompute_or_load_feats(data, args.data_path, args)
            tupla = PandoraAttGen.generate_setup(data, df, args.tasktype, feat_names, args, extra_feats, extra_feat_names, label_name, feat_size=15784, hp=8, fold_in=4)
            PandoraAttGen.model_dic[label_name] = PandoraSetUp(tupla[0], tupla[1], tupla[2])
        return None

    @staticmethod
    def __init_age_model():
        argstr = '-data_path C:\\Users\\konom\\Downloads\\pandora_baseline\\data -label age -tasktype regression -folds age -feats 1gram -model lr -variant LR-N'
        args = PandoraAttGen.parse_args(argstr.split())
        unm = pickle.load(open(os.path.join(args.data_path, "unames.pickle"), "rb"))
        txt = []
        labels = ["age"]
        for label_name in labels:
            data = PandoraAttGen.load_data(unm, txt, os.path.join(args.data_path, "author_profiles.csv"), label_name, args.tasktype, args.folds, 0, args)
            df, feat_names, extra_feats, extra_feat_names = PandoraAttGen.precompute_or_load_feats(data, args.data_path, args)
            tupla = PandoraAttGen.generate_setup(data, df, args.tasktype, feat_names, args, extra_feats, extra_feat_names, label_name, feat_size=15784, hp=8, fold_in=4)
            PandoraAttGen.model_dic[label_name] = PandoraSetUp(tupla[0], tupla[1], tupla[2])
        return None

    @staticmethod
    def __init_gender_model():
        argstr = '-data_path C:\\Users\\konom\\Downloads\\pandora_baseline\\data -label is_female -tasktype classification -folds gender -feats 1gram -model lr -variant LR-N'
        args = PandoraAttGen.parse_args(argstr.split())
        unm = pickle.load(open(os.path.join(args.data_path, "unames.pickle"), "rb"))
        txt = []
        labels = ["is_female"]
        for label_name in labels:
            data = PandoraAttGen.load_data(unm, txt, os.path.join(args.data_path, "author_profiles.csv"), label_name, args.tasktype, args.folds, 0, args)
            df, feat_names, extra_feats, extra_feat_names = PandoraAttGen.precompute_or_load_feats(data, args.data_path, args)
            tupla = PandoraAttGen.generate_setup(data, df, args.tasktype, feat_names, args, extra_feats, extra_feat_names, label_name, feat_size=15784, hp=8, fold_in=4)
            PandoraAttGen.model_dic[label_name] = PandoraSetUp(tupla[0], tupla[1], tupla[2])
        return None

    @staticmethod
    def load_data(unames, texts, labels_path, label_name, label_type, fold_grp, repeat,
                  args):  # percentiles or scores THAT PARAMETER IS DEPRECATED DOES NOTHING NOW
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

                PandoraAttGen.feat_names = gram_feat_names

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

    @staticmethod
    def generate_setup(data, data_feats, label_type, feat_names, args, extra_feats,
                       extra_feat_names, label_name, feat_size=None, hp=None, fold_in=None):
        # important return stuf
        xval_res = None
        tfidf = None
        fs = None

        cw = "balanced"  # "balanced" or None
        # reg_types = ["l1","l2"]
        reg_types = [
            "l2"]  # preliminary experiments show that l2 + relatively low number of features in feat sel perform the best, but this might not always be the case

        n_base_feats = data_feats.shape[1] if data_feats is not None else 0
        n_extra_feats = extra_feats.shape[1] if extra_feats is not None else 0
        total_n_feats = n_base_feats + n_extra_feats

        max_features = 20000
        feat_sel_Ncandidates = [int(percentage * total_n_feats) for percentage in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]]
        if total_n_feats < max_features:
            feat_sel_Ncandidates += ["all"]  # if there are not a lot of feats also try a variant with all feats
        else:  # on the other hand, dont try more than 20k feats (more than that didn't appear to yield significant benefits in prelim. experiments)
            feat_sel_Ncandidates = [x for x in feat_sel_Ncandidates if x <= max_features] + [max_features]

        if feat_size:
            feat_sel_Ncandidates = [feat_size]

        if args.model == "lr":
            hyperparams = [tuple([2 ** i]) for i in range(-10, 5)]  # regularisation strength for the regression models
        elif args.model == "et":
            n_estimators = [100, 200, 300, 400, 500]
            mf = ["auto", 800] if total_n_feats >= 800 else ["auto"]
            bootstrap = [True]
            oob = [True, False]
            hyperparams = list(np.itertools.product(*[n_estimators, mf, bootstrap, oob]))
            feat_sel_Ncandidates = ["all"] if total_n_feats < max_features else [max_features]
        elif args.model == "dummy":
            hyperparams = [0]
        else:
            raise Exception("Unknown model type: " + str(args.model))

        if hp:
            hyperparams = [(hp,)]

        valid_indexes = data['label'].notnull()

        # filter out rows with nan vals for extra feats
        if extra_feats is not None:
            ll = np.isnan(extra_feats.todense()).any(axis=1)
            for ind in range(len(ll)):
                if ll[ind]:  # there is a nan in that row
                    valid_indexes[ind] = False
            print("Threw out " + str(np.sum(ll)))

        valid_indexes_numbers = np.where(valid_indexes)[0]

        filtered_data = data[valid_indexes]
        if data_feats is not None:
            filtered_data_feats = data_feats[valid_indexes_numbers, :]
        if extra_feats is not None:
            filtered_extra_feats = extra_feats[valid_indexes_numbers, :]

        folds_to_run = [0, 1, 2, 3, 4] if args.specificfold == -1 else [args.specificfold]
        if fold_in:
            folds_to_run = [fold_in]

        for fold in folds_to_run:
            # print("Starting fold " + str(fold))

            test_fold = fold
            val_fold = (fold + 1) % 5

            train_indexes = (filtered_data['fold'] != test_fold) & (filtered_data['fold'] != val_fold)
            train_indexes_numbers = np.where(train_indexes)[0]

            if data_feats is not None:
                train_feats = filtered_data_feats[train_indexes_numbers]

            if extra_feats is not None:
                train_extra_feats = filtered_extra_feats[train_indexes_numbers]

            train_labels = filtered_data[train_indexes]["label"]

            # apply tfidf weighting
            if data_feats is not None:
                # print("Applying tfidf for this fold.")
                tfidf = TfidfTransformer(sublinear_tf=True)
                train_feats = tfidf.fit_transform(train_feats)

            train_unames = list(filtered_data[train_indexes]["author"])

            # some fixes on the extra feats part
            if extra_feats is not None:
                # scaler = StandardScaler(with_mean = False)
                scaler = MinMaxScaler()

                train_extra_feats = csr_matrix(scaler.fit_transform(train_extra_feats.todense()))

            # combine word feats with all the other feats
            if data_feats is not None and extra_feats is None:
                combined_train_feats = train_feats
            elif data_feats is None and extra_feats is not None:
                combined_train_feats = csr_matrix(train_extra_feats)
            elif data_feats is not None and extra_feats is not None:
                for i in range(train_extra_feats.shape[0]):
                    if np.isnan(train_extra_feats.todense()[i, :]).any():
                        print("NAN FOUND FOR USER :" + train_unames[i])

                combined_train_feats = csr_matrix(sparse_hstack([train_feats, csr_matrix(train_extra_feats)]))
            else:
                raise Exception("You must supply at least one type of features to use!")

            # run the many loops for testing various versions of this and that
            for feats_N in feat_sel_Ncandidates:
                fs = SelectKBest(chi2, k=feats_N) if label_type == "classification" else SelectKBest(f_regression,
                                                                                                     k=feats_N)
                if (feats_N == 0):
                    continue
                train_feats_FS = csr_matrix(fs.fit_transform(combined_train_feats, train_labels))

                def eval_hp(hype, r, l, c, ar, trf, trl):
                    model = PandoraAttGen.spawn_model(hype, r, l, c, ar)
                    model.fit(trf, trl)
                    # print("Finished for " + str(hype))
                    return model

                for reg in reg_types:
                    train_feats_FS.sort_indices()

                    xval_res = Parallel(n_jobs=12)(
                        delayed(eval_hp)(h, reg, label_type, cw, args, train_feats_FS, train_labels) for h in
                        hyperparams)

        print("*")
        print("*")
        print("*")
        print("*")

        return xval_res[0], fs, tfidf

    @staticmethod
    def spawn_model(hyperparam_combo, reg, label_type, cw, args):
        if label_type == "classification":
            # model = LinearSVC(C = C, penalty = reg, max_iter = 5000, class_weight = cw)
            if args.model == "lr":
                model = LogisticRegression(C=hyperparam_combo[0], penalty=reg, class_weight=cw, max_iter=20000)
            elif args.model == "et":
                model = ExtraTreesClassifier(n_estimators=hyperparam_combo[0], max_features=hyperparam_combo[1],
                                             bootstrap=hyperparam_combo[2], oob_score=hyperparam_combo[3], n_jobs=-1,
                                             class_weight=cw)
            elif args.model == "dummy":
                # model = DummyClassifier(strategy = "most_frequent")
                model = DummyClassifier(strategy="stratified")
            else:
                raise Exception("Unknown model type: " + str(args.model))
            # model = DummyClassifier(strategy="most_frequent")
        elif label_type == "regression":
            if args.model == "lr":
                if reg == "l1":
                    model = Lasso(alpha=hyperparam_combo[0], max_iter=20000, solver='sag')
                elif reg == "l2":
                    model = Ridge(alpha=hyperparam_combo[0], max_iter=20000, solver='sag')
                else:
                    raise Exception("Unkown regularisation type -- " + str(reg))
            elif args.model == "et":
                model = ExtraTreesRegressor(n_estimators=hyperparam_combo[0], max_features=hyperparam_combo[1],
                                            bootstrap=hyperparam_combo[2], oob_score=hyperparam_combo[3], n_jobs=-1)
            elif args.model == "dummy":
                model = DummyRegressor(strategy="mean")
            else:
                raise Exception("Unknown model type: " + str(args.model))
        else:
            raise Exception("Unkown label type -- " + str(label_type))
        return model
