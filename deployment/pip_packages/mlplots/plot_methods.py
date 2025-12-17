import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import matplotlib.cm as cm
import pandas as pd
from mlplots import setup_logger
logger = logging.getLogger('mlplots.plotmethods')
logger.setLevel(logging.DEBUG)


class ShaplyPlot:

    def __init__(self, shaps, features, feature_names, cutoff=5, subsample=True):
        self.n_features = len(feature_names)
        if len(feature_names) != self.n_features:
            raise ValueError('feature names.txt is not the same lengths as n_cols in features')
        if subsample:
            if self.n_features > 10000:
                # grab random 10000
                index = np.random.randint(0, self.n_features + 1, 10000)
                self.features = features[index, :]
                self.shaps = shaps[index, np.arange(self.n_features)]
            else:
                self.features = features
                self.shaps = shaps[:, np.arange(self.n_features)]

        self.features_names = feature_names

        self.cutoff = cutoff

        self.params = {'cmap': 'RdYlBu_r',
                       'alpha': 1,
                       'edgecolors': 'k',
                       'lw': .2,
                       'linewidth': 0}

    def plot_columns(self, col_name, num_features=12):
        '''
        Plots a series of
        :param col_name: regex to search for columns
        :return:
        '''
        regex = '.*' + col_name + '.*'
        col_idx = [i for i, fn in enumerate(self.features_names) if re.search(regex, fn)]
        if not col_idx:
            raise ValueError(col_name + '  not in found in feature names.txt')
        shaps = np.squeeze(self.shaps[:, col_idx])
        f_name = np.array(self.features_names)[col_idx]
        select_shaps = np.array([np.take(shaps, i, axis=1) for i, col_name in enumerate(f_name[0:-1])])

        mean_shaps = np.array(
            [np.mean(np.abs(np.take(select_shaps, i, axis=1))) for i, col_name in enumerate(f_name[0:-1])])
        median_shaps = np.array(
            [np.median(np.take(select_shaps, i, axis=1)) for i, col_name in enumerate(f_name[0:-1])])
        indexmean = np.argsort(mean_shaps)
        index = np.argsort(median_shaps)

        index = index[0:num_features]
        sorted_mean = mean_shaps[index]
        sorted_median = median_shaps[index]

        bp = plt.boxplot(select_shaps[index].tolist(), labels=f_name[index], vert=False, meanline=False,
                         showmeans=False, meanprops=dict(linestyle='-', color='black'),
                         medianprops=dict(color='black'), patch_artist=True, showfliers=False)
        max, min = np.max(select_shaps), np.min(select_shaps)

        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        cmap = matplotlib.cm.get_cmap('RdYlBu_r')
        # cmap = plt.cm.get_cmap('RdBu', 6)

        for i, patch in enumerate(bp['boxes']):
            color = cmap(norm(sorted_median[i]))[0:3]
            # print("patch")
            # print("value ", sorted_median[i])
            # print("color is: ", color)
            patch.set(facecolor=color)

        # formatting boundary lines to only display left and bottom boundaries
        spines = plt.gca().spines.values()
        for i, spine in enumerate(plt.gca().spines.values()):
            if (i != 0 and i != 2):
                spine.set_visible(False)
        plt.xlim((-.1, .1))
        plt.show()

    def _plot_helper(self, feature_vals, shap_vals, feature_name, s=2, ylim=None, xlim=None, facecolor='dimgrey'):
        '''
        Creates the shapley plot using matplotlib
        :param feature_vals: x values
        :param shap_vals: y values
        :param feature_name: label on graph
        :param s: size of scatter plot points
        :param ylim: user specified or calculated in helper function self._xlim_spacer
        :param xlim: user specified or calculated in helper function self._ylim
        :param facecolor: background color of plot
        :return:
        '''

        fig, ax = plt.subplots()
        ax.set_facecolor(facecolor)

        # Add gridlines
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray')
        ax.xaxis.grid(color='gray')

        # create scatter plot
        plt.scatter(feature_vals, shap_vals, c=shap_vals, vmin=ylim[0], vmax=ylim[1],
                    rasterized=len(feature_vals) > 500,
                    **self.params)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.colorbar()
        plt.gcf().set_size_inches(6, 5)
        plt.xlabel(feature_name, fontsize=13)
        plt.ylabel("SHAP value for\n" + feature_name, fontsize=13)
        plt.axhline(y=0.0, color='k', linestyle='-')
        plt.title(feature_name, fontsize=13)
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().tick_params(labelsize=11)

        for spine in plt.gca().spines.values():
            spine.set_edgecolor("#333333")

        # rotate x tick labels so they don't overlap
        if max(xlim) >= 10000:
            rotate = 40
        else:
            rotate = 0
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=rotate)
        logger.debug('calling plt.show...')

        # show ax
        plt.show(ax)
        logger.debug('plt.show(ax) called')

    ## jitter helper functions ##
    def _jitter_features(self, features, jitter):
        jitter_features = np.random.normal(0, jitter, len(features))
        return jitter_features

    def _jitter_shaps(self, shaps, jitter):
        jitter_shaps = np.random.normal(0, jitter, len(shaps))
        return jitter_shaps

    # jitter added to X and Y
    def _jitter_x_y(self, features, shaps, u_index):
        jitter = len(u_index) * .001  # jitter = number of unique indeces * .001
        features = features + self._jitter_features(features, jitter)
        shaps = shaps + self._jitter_shaps(shaps, jitter)
        logger.debug('jitter added to x AND y : {}'.format(jitter))
        return features, shaps

    # jitter added to X
    def _jitter_x(self, features, u_index, notu_index):
        jitter = len(u_index) * .01  # jitter = number of unique indeces * .01
        jitter_features = self._jitter_features(features, jitter)
        features[notu_index] = features[notu_index] + jitter_features[notu_index]
        logger.debug('jitter added to X : {}'.format(jitter))
        return features

    # one hot encoded jitter user specified
    def _onehot_jitter(self, features, jitter):
        features = features + self._jitter_features(features, jitter)
        logger.debug('onehot jitter added by user: {}'.format(jitter))
        return features

    # x integer jitter
    def _int_jitter(self, features, shaps):
        unique = len(np.unique(features))
        if unique < self.cutoff:
            num = unique * .001
            logger.debug('one hot jitter')
        else:
            num = .001  # Change if you want more or less jitter
            logger.debug('integer jitter')
        jitter = unique * num
        features = features + self._jitter_features(features, jitter)
        shaps = shaps + self._jitter_shaps(shaps, jitter)
        logger.debug('jitter X and Y: {}'.format(jitter))
        return features, shaps

    # user specified jitter
    def _user_jitter(self, features, shaps, jitter):
        features = features + self._jitter_features(features, jitter)  # add noise to X axis
        shaps = shaps + self._jitter_shaps(shaps, jitter)  # add noise to y axis
        logger.debug('jitter added by user : {}'.format(jitter))
        return features, shaps

    # set xlim bounds (non categorical)
    def _xlim_spacer(self, features, vals):
        xlim = np.percentile(features, vals)
        x = xlim.copy()
        lower_bound = np.percentile(x, 5)
        upper_bound = np.percentile(x, 95)
        space = np.absolute(lower_bound)

        # subtract space from left side
        if (np.percentile(features, 5) <= lower_bound):
            xlim[0] = x[0] - space
        else:
            pass

        # add space to right side
        if (np.percentile(features, 95) >= upper_bound):
            xlim[1] = x[1] + space
        else:
            pass

        logger.debug('xlim original : {}'.format(x))
        logger.debug('new xlim : {}'.format(xlim))
        return xlim

    # set ylim bounds
    def _ylim(self, shaps):
        y = [np.min(shaps), np.max(shaps)]
        space = np.absolute(np.percentile(y, 10))
        ylim = [y[0] - space, y[1] + space]

        if ylim[0] < -2:
            ylim[0] = -2
        if ylim[1] > 2:
            ylim[1] = 2

        logger.debug('ylim no space : {}'.format(y))
        logger.debug('new ylim : {}'.format(ylim))
        return ylim

    # xlim if categorical
    def _xlim_cat(self, features):
        space = len(np.unique(features)) * 0.05
        xlim = (np.min(features) - space, np.max(features) + space)
        logger.debug('xlim cat : {}'.format(xlim))
        return xlim

    # jitter if categorical
    def _jitter_cat(self, features):
        unique = len(np.unique(features))
        num = unique * .002
        jitter = unique * num
        features = features + self._jitter_features(features, jitter)
        logger.debug('cat jitter X : {}'.format(jitter))
        return features

    def plot(self, col_name, jitter=None, s=2, xlim=None, ylim=None, facecolor='dimgrey', cat=False, vals=(1, 99)):
        '''
        Removes outliers from features, determines appropriate xlim and ylim.
        Determines jitter amount if none specified. Calls helper function self._plot_helper.
        :param col_name: used to find feature and shap values
        :param jitter: amount of noise added to x and y values of plot
        :param s: size of scatter plot points
        :param xlim: x axis of scatter plot
        :param ylim: y axis of scatter polot
        :param facecolor:
        :param cat: whether data is categorical. default set to False
        :param vals: determines what percentile of features to use in plot. default [1, 99]
        :return:
        '''
        if col_name not in self.features_names:
            raise ValueError(col_name + '  not in found in feature names.txt')
        k = [i for i, j in enumerate(self.features_names) if j == col_name][0]

        features = np.squeeze(self.features[:, k])
        shaps = self.shaps[:, k]
        f_name = self.features_names[k]

        # check if categorical
        if len(np.unique(features)) < self.cutoff:
            cat = True

        # set xlim if none
        if xlim is None:
            if cat is True:
                xlim = self._xlim_cat(features)
            elif cat is not True:
                xlim = self._xlim_spacer(features, vals)
        else:
            pass

        # y axis if None
        if ylim is None:
            ylim = self._ylim(shaps)
        else:
            pass

            arr = np.concatenate((np.rint(shaps), np.rint(features)), axis=0)  # [shaps],[features] ints
            unique_arr, u_index, notu_index = np.unique(arr, return_index=True, return_inverse=True)

            unique_features = len(np.unique(features))  # number of unique features
            unique_index = len(u_index)  # number of unique indeces

            # add jitter
            if cat is False:
                if jitter is not None:
                    features, shaps = self._user_jitter(features, shaps, jitter)
                # x are ints, jitter x and y
                elif unique_index < 100 and unique_features < 100:
                    features, shaps = self._int_jitter(features, shaps)

                # x jitter - certain areas
                elif unique_index >= 100 and unique_index < len(features) / 4:
                    features = self._jitter_x(features, u_index, notu_index)
                else:
                    pass

            # add jitter for categorical
            if cat is True:
                if jitter is None:
                    features = self._jitter_cat(features)
                # add user specified jitter
                elif jitter is not None:
                    features = self._onehot_jitter(features, jitter)
                else:
                    pass

        self._plot_helper(features, shaps, f_name, s=s, xlim=xlim, ylim=ylim, facecolor=facecolor)


class ClassificationPlot:
    def __init__(self, train, test=None, eval=None, cutoff=None, key='acc', subsample=True):
        import numpy as np

        self.data_dict = {
            "train": train
        }
        if test is not None:
            self.data_dict["test"] = test
        if eval is not None:
            self.data_dict["eval"] = eval

        for k, item in self.data_dict.items():
            labels = item[0]
            preds = item[1]
            if len(labels) != len(preds):
                raise ValueError('len preds : {0} != len labels : {1}'.format(len(preds), len(labels)))
            if subsample:
                if len(labels) > 10000:
                    # grab random 10000
                    index = np.random.randint(0, len(labels) + 1, 10000)
                    self.data_dict[k] = np.array((labels[index], preds[index]))

        self.labels = self.data_dict.get("train")[0]
        self.preds = self.data_dict.get("train")[1]
        if test is not None:
            self.data_dict["test"] = test
            self.test_labels = self.data_dict.get("test")[0]
            self.test_preds = self.data_dict.get("test")[1]
        else:
            self.test_performance = None
        if eval is not None:
            self.data_dict["eval"] = eval
            self.eval_preds = self.data_dict.get("eval")[0]
            self.eval_labels = self.data_dict.get("eval")[1]


        for k, item in self.data_dict.items():
            labels = item[0]
            preds = item[1]
            if len(labels) != len(preds):
                raise ValueError(' len preds : {0} != len labels : {1} '.format(len(preds), len(labels)))
            if subsample:
                if len(labels) > 10000:
                    # grab random 10000
                    index = np.random.randint(0, len(labels), 10000)
                    self.data_dict[k] = np.array((labels[index], preds[index]))

        self.update_cutoff(cutoff, key=key)

    def update_cutoff(self, cutoff=None, key=None):
        from FeaturePipe.utils import class_performance, optimum_cutoff
        if cutoff is None:
            self.perf_dict = optimum_cutoff(self.labels, self.preds, key=key)
            self.cutoff = self.perf_dict['best_cutoff']
        if key is None:
            self.key = 'acc'
        else:
            self.key = key
        self.train_performance = class_performance(self.labels, self.preds, self.cutoff)

    def plot_sense_spec(self):
        '''

        :return:
        '''
        from sklearn.metrics import roc_curve
        import pandas as pd
        import numpy as np
        from matplotlib import pyplot as plt
        fpr, tpr, thresholds = roc_curve(self.labels, self.preds)
        # print("Area under the ROC curve : %f" % roc_auc)
        i = np.arange(len(tpr))  # index for df
        roc = pd.DataFrame(
            {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
             'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})

        # Plot tpr vs 1-fpr
        fig, ax = plt.subplots()
        plt.plot(roc['tpr'], color='orange', label='sensitivity')
        plt.plot(roc['1-fpr'], color='blue', label='specifity')
        plt.xlabel('Cut Off')
        plt.ylabel('Sensitiviy | Specificity')
        plt.title('Sensitivity, Specificity vs Cut Off')
        ax.set_xticklabels([])
        plt.legend()
        plt.show()

    def plot_auc(self):
        '''

        :return:
        '''
        from sklearn.metrics import roc_auc_score, roc_curve
        import numpy as np
        import matplotlib.pyplot as plt
        np.random.seed(100)
        plt.figure(0).clf()

        color_scheme = {'train': 'blue', 'test': 'orange', 'eval': 'orange'}
        for k, item in self.data_dict.items():
            fpr, tpr, thresh = roc_curve(item[0], item[1])
            auc = roc_auc_score(item[0], item[1])
            plt.plot(fpr, tpr, color=color_scheme.get(k), label=k + " auc: " + str(round(auc, 3)))

        # mlplots a base line
        label = np.random.choice([0, 1], 500, replace=True)
        pred = np.random.normal(1, 1, 500)
        fpr, tpr, thresh = roc_curve(label, pred)
        auc = roc_auc_score(label, pred)
        plt.plot(fpr, tpr, linestyle='dashed', color='darkblue', label="Random auc:" + str(round(auc, 3)))
        plt.legend(loc=0)
        plt.show()

    def plot_performance_bars(self, bars=('tpr', 'tnr', 'ppv', 'npv'), width=.25):
        '''
        :param cutoff:
        :param optimize_key:
        :return:
        '''
        import matplotlib.pyplot as plt
        from FeaturePipe.utils import class_performance
        import numpy as np

        cutoff = self.cutoff
        key = self.key
        fig, ax = plt.subplots()

        bar_offset = 0
        for k, item in self.data_dict.items():
            labels = item[0]
            preds = item[1]
            plot_label = k.capitalize() + ' Set'
            performance = class_performance(labels, preds, cut_off=cutoff)
            rects = ax.bar(np.arange(len(bars)) + bar_offset, [performance[b] for b in bars], width, label=plot_label)
            self.add_bar_label(rects, [performance[b] for b in bars])
            bar_offset += width

        plt.xlabel('Metric', fontsize=9)
        plt.ylabel('Percentage', fontsize=9)
        plt.title('Model Metrics Using Optimal {0} Cutoff: {1}'.format(key, round(cutoff, 3)), fontsize=11)
        plt.yticks(fontsize=6)
        plt.xticks(np.arange(4) + width / 2, bars, fontsize=8)
        plt.ylim(0, 1.08)
        plt.legend(loc=2, prop={'size': 8})
        plt.show()

    def add_bar_label(self, rects, labels, horizontal=False):
        import matplotlib.pyplot as plt
        # For each bar: Place a label
        for index, rect in enumerate(rects):
            width = rect.get_width()
            height = rect.get_height()
            color = 'black'
            if horizontal:
                xloc = width + .03
                yloc = rect.get_y() + height / 2
                plt.text(xloc, yloc, labels[index], verticalalignment='center', color=color)
                mid_xloc = width / 2
                plt.text(mid_xloc, yloc, "{:.3f}".format(width), verticalalignment='center',
                         horizontalalignment='center')
            else:
                xloc = rect.get_x() + width / 2
                label = "{:.3f}".format(labels[index])
                plt.annotate(label, (xloc, height), xytext=(0, 1), textcoords="offset points", ha='center', fontsize=8)

    def plot_groups(self, groups, key, data_set='train'):
        '''
            #groups is a list corresponding to labels - most likely use case is department
        :param groups: ['ICU', 'Neuro', 'NEONAT', 'ICU', 'Surgical Trauma', 'Neuro', 'ICU', 'Cardiovascular']
        :param key: 'auc'
        :return:
        '''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from FeaturePipe.utils import class_performance

        dataset = self.data_dict.get(data_set)
        labels = dataset[0]
        preds = dataset[1]
        if len(groups) != len(labels):
            raise ValueError('groups is not the same length as labels')

        fig, ax = plt.subplots()

        num_groups = len(np.unique(np.array(groups)))

        df = pd.DataFrame({'Group': np.array(groups), 'Labels': np.array(labels), 'Preds': np.array(preds)})
        performances = []
        for group in df['Group'].unique():
            print('-----------------------')
            subgroup = df.groupby(['Group'])
            #print(subgroup.get_group(group))
            subgroup_labels = subgroup.get_group(group)['Labels'].values
            subgroup_preds = subgroup.get_group(group)['Preds'].values
            performance = class_performance(subgroup_labels, subgroup_preds, self.cutoff)
            #print('Performance for group ', group, ' is: ', performance[key])
            performances.append(performance[key])

        rects = ax.barh(np.arange(num_groups), performances, align='center', tick_label='')
        self.add_bar_label(rects, np.unique(np.array(groups)), horizontal=True)
        max = np.max(labels)
        xlim = max * .6 + max
        plt.xlim(0, xlim)
        plt.ylabel('Groups', fontsize=9)
        plt.xlabel('Percentage', fontsize=9)
        plt.title('Metrics by Group Using Optimal {0} Cutoff: {1}'.format(key,round(self.cutoff, 3)), fontsize=11)
        plt.yticks(fontsize=6)
        plt.show()

    def plot_cutoff_metric(self, cutoff_bounds=(0, 1), key=None):
        '''

        :param cutoff_bounds:
        :param optimize_key:
        :return:
        '''
        # tom updated
        import matplotlib.pyplot as plt
        import numpy as np
        from FeaturePipe.utils import optimum_cutoff

        plt.figure(0).clf()
        if key is None:
            key = 'acc'

        for k, item in self.data_dict.items():
            labels = item[0]
            preds = item[1]
            optimums = optimum_cutoff(labels=labels, preds=preds, key=key, cut_range=cutoff_bounds, step=0.1)
            performance = optimums['performance']
            cutoff = optimums['cut_offs']
            if k == 'train':
                max_performance = np.max(performance)
                max_performance_cutoff = optimums['best_cutoff']
            plt.plot(cutoff, performance, label=k.capitalize(), lw=3, zorder=-2)

        plt.legend()
        plt.scatter(max_performance_cutoff, max_performance, c='k', zorder=1)
        plt.axvline(x=max_performance_cutoff, color='k')
        plt.gcf().set_size_inches(6, 5)
        plt.xlabel('Cutoff Values, best value: {0}'.format(round(max_performance_cutoff, 2)), fontsize=13)
        plt.ylabel("Model %s" % key, fontsize=13)
        plt.title('Optimum Cutoff: %s' % key, fontsize=13)
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().tick_params(labelsize=11)
        for spine in plt.gca().spines.values():
            spine.set_edgecolor("#333333")
        plt.show()
        return round(max_performance_cutoff, 2)

    def plot_confusion_matrix(self):
        import seaborn as sn
        import pandas as pd
        import matplotlib.pyplot as plt
        best_cutoff = self.cutoff
        d = self.perf_dict
        key = self.key
        tp = d['all_measures']['tp']
        fp = d['all_measures']['fp']
        fn = d['all_measures']['fn']
        tn = d['all_measures']['tn']
        conf_mat = [[tp, fn], [fp, tn]]
        conf_mat = pd.DataFrame(conf_mat, index=['Actually  1', 'Actually 0'],
                                columns=['Predicted 1', 'Predicted 0'])

        title = 'Confusion Matrix with best ' + key + ' cutoff: ' + str(round(best_cutoff, 3))
        plt.figure(0).clf()
        sn.heatmap(conf_mat, annot=True, fmt="d", ).set_title(title)
        plt.show()


    def plot_stick_man(self, **kwargs):
        # params = {'x_axis': 10,
        #          'scale': 100,
        #          'zoom': .03,
        #          'sectioned': False,
        #          'colors': ('b', 'g', 'r', 'c',)}
        # params.update(**kwargs)
        # p = PlotStickMan(self.labels, self.preds, self.cutoff)
        # p.plot_stick_man(*params)
        raise NotImplemented


class PlotStickMan():
    def __init__(self, y_true, preds, cutoff=.5):
        self.y_true = y_true
        self.preds = preds
        self.cutoff = cutoff

    def get_error_amounts(self):
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for y_true, y_pred in zip(self.y_true, self.preds):
            positive = y_pred >= self.cutoff
            if y_true == 1 and positive:
                true_pos += 1
            if y_true == 0 and positive:
                false_pos += 1
            if y_true == 1 and not positive:
                true_neg += 1
            if y_true == 0 and not positive:
                false_neg += 1
        return true_pos, false_pos, true_neg, false_neg

    def get_y_axis(self, true_pos, false_pos, true_neg, false_neg, max_x, sectioned):
        vals = [true_pos, false_pos, true_neg, false_neg]
        y_axis = 0
        if not sectioned:
            total = sum(vals)
            y_axis = total//max_x
            if total % max_x > 0:
                y_axis+=1
        elif sectioned:
            y_axis = 0
            for val in vals:
                y_axis += val // max_x
                if val % max_x > 0:
                    y_axis += 1
        return y_axis

    def change_rgba(self, img, rgba):
        new_img = img
        # png resources stored as nested arrays containing rgba vals (may not work with other img types)
        for i in range(len(new_img)):
            for j in range(len(new_img[i])):
                # finds non-white(0 alpha) and changes to new rgba
                if new_img[i][j][3] > 0:
                    new_img[i][j] = rgba

        return new_img

    def get_scaled(self, true_pos, false_pos, true_neg, false_neg, scale):
        vals = [true_pos, false_pos, true_neg, false_neg]
        total = sum(vals)
        # calculate scaled values
        for i in range(len(vals)):
            vals[i] = round(vals[i] * scale / total)
        return vals

    def plot_stick_man(self, x_axis=10, scale=100, zoom=.03, sectioned=False, colors=('b', 'g', 'r', 'c',)):
        '''
        :param y_trues: true vals for y
        :param y_preds: predicted y, pre-cutoff
        :param cutoff: threshold for true/false
        :param x_axis: individual points along the x axis
        :param scale: scale y amounts
        :param zoom: size of image for stickman
        :param sectioned: section rates into different rows
        :param colors: colors in (true pos, false pos, true neg, false neg)
        :param spacing: spacing between stickmen
        :return: displays graph of stickmen
        '''
        import pkg_resources
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        ax = plt.subplot()

        true_pos, false_pos, true_neg, false_neg = self.get_error_amounts()
        if scale is not None:
            true_pos, false_pos, true_neg, false_neg = self.get_scaled(true_pos, false_pos, true_neg, false_neg, scale)

        y_axis = int(self.get_y_axis(true_pos, false_pos, true_neg, false_neg, x_axis, sectioned))
        ax.axis([0, x_axis-1, 0, y_axis])

        if pkg_resources.resource_exists('rnner', 'resources/stickman.png'):
            img_data = pkg_resources.resource_stream('rnner', 'resources/stickman.png')
            img = plt.imread(img_data)
        else:
            print('File Not Found')
            return

        # generates list of img of different, specified colors
        stickmen = [OffsetImage(self.change_rgba(img, mcolors.to_rgba_array(colors[i])), zoom=zoom) for i in
                    range(len(colors))]

        vals = [true_pos, false_pos, true_neg, false_neg]

        # don't count rates of 0
        tmp = vals.copy()
        for i in range(len(vals)):
            if tmp[i] == 0:
                vals.pop(i)
                stickmen.pop(i)

        # counter for deciding image/row changes
        i = 0
        for y in range(y_axis, 0, -1):
            for x in range(0, x_axis):
                xy = (x, y)
                if not i < len(vals):
                    break
                elif vals[i] > 0:
                    # individually plotting resources
                    abb = AnnotationBbox(stickmen[i], xy, bboxprops=dict(ec='w'))
                    ax.add_artist(abb)
                    vals[i] -= 1
                elif vals[i] == 0 and sectioned:
                    y -= 1
                    i += 1
                    break
                elif vals[i] == 0 and not sectioned:
                    i += 1
                    if i < len(vals):
                        # allows continuation of resources as the values change
                        abb = AnnotationBbox(stickmen[i], xy, bboxprops=dict(ec='w'))
                        ax.add_artist(abb)
                        vals[i] -= 1

        # plotting legend/formatting axis
        texts = ["True Positive", "False Positive", "True Negative", "False Negative"]
        patches = [mpatches.Patch(color=colors[i], label=texts[i]) for i in range(len(texts))]
        ax.legend(handles=patches, bbox_to_anchor=(0, -.1, 1, .1), loc=9, mode="expand", ncol=4, prop={'size': 8})
        ax.axis('off')
        plt.show()


def plot_density(preds=None, true=None, labels=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    fig, ax = plt.subplots(figsize=(4, 4))
    preds = np.array(preds)
    true = np.array(true)
    if preds.ndim == 1:
        preds = np.reshape(preds, (preds.shape[0], 1))
    if true.ndim == 1:
        true = np.reshape(true, (true.shape[0], 1))
    for i in range(preds.shape[1]):
        m = np.mean(preds[:, i])
        label = ' Mean Prob:  %0.2f' % m
        if labels is not None:
            label = labels[i] + label
        sns.distplot(preds[:, i], hist=False, label=label)
        if true is not None:
            m = np.mean(true[:, i])
            label = ' True Prob: %0.2f' % m
            if labels is not None:
                label = labels[i] + label
            plt.axvline(m, color='r', label=label)

    plt.title("Distribution of Predicted Probabilities", loc='right')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend(loc="right")
    plt.show()


class Interaction:

    def __init__(self, features,  feature_names, xgb_model=None, predict_fun=None, **kwargs):
        '''

        :param features: np array of features
        :param predict_fun: function that uses a predict method, applied to features, returning 1d output
        :param feature_names: list of feature names
        :param kwargs {'x': (1, 99), 'y': (1, 99), 'steps': 100}
        limits for the x and y axis, steps are the number of steps used when creating the mesh grid
        '''
        logger.debug('Interaction plot method init')
        self.feature_names = feature_names
        self.predict_fun = predict_fun
        self.features = features
        self.model = xgb_model
        n_features = len(feature_names)
        if n_features == features.shape[1]:
            logger.debug('interaction feature name len matches features.shape[1]')
        else:
            logger.error('n_feature names {0} miss matches features shape[1] {1}'.format(n_features ,features.shape[1]))
        self.params = {'x': (1, 99), 'y': (1, 99), 'steps': 20}
        self.params.update(**kwargs)

    def plot(self, x_name, y_name, unique_x=False, unique_y=False,  **kwargs):
        '''

        :param x_name: str found in feature names (x axis)
        :param y_name: str found in feature names (y axis)
        :param kwargs: {'x': (1, 99), 'y': (1, 99), 'steps': 100}
        :return:
        '''
        from matplotlib import pyplot as plt
        self.params.update(**kwargs)
        logger.debug('Interaction plot method called')
        logger.debug('interaction plot params {0}'.format(self.params))
        missing = set([x_name, y_name]).difference(set(self.feature_names))
        if missing:
            msg = 'input columns {0} missing from feature_names'.format(missing)
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.debug('input cols to Interaction plot validated')
        X = self.features

        # get index positions of the input names
        index_x = [i for i, j in enumerate(self.feature_names) if j == x_name][0]
        index_y = [i for i, j in enumerate(self.feature_names) if j == y_name][0]

        steps = self.params['steps']
        # get max and min values for the feature
        if unique_x:
            new_x_vals = list(set(X[:, index_x]))
            logger.debug('Interaction Plot method with n unique x vals {0}'.format(len(new_x_vals)))
        else:
            max_val_x = np.percentile(X[:, index_x], self.params['x'][1])
            min_val_x = np.percentile(X[:, index_x], self.params['x'][0])
            logger.debug(' {0} min, {1} max x vals'.format(min_val_x, max_val_x))
            new_x_vals = np.arange(min_val_x, max_val_x, (max_val_x - min_val_x) / steps)

        if unique_y:
            new_y_vals = list(set(X[:, index_y]))
            logger.debug('Interaction Plot method with n unique y vals {0}'.format(len(new_y_vals)))
            # get max and min values for the feature
        else:
            max_val_y = np.percentile(X[:, index_y], self.params['y'][1])
            min_val_y = np.percentile(X[:, index_y], self.params['y'][0])

            logger.debug(' {0} min, {1} max y vals'.format(min_val_y, max_val_y))

            # create a grid of every posible combination of xy
            steps = self.params['steps']
            logger.debug('creating mesh grid with steps {0}'.format(steps))
            new_y_vals = np.arange(min_val_y, max_val_y, (max_val_y - min_val_y) / steps)

        xx, yy = np.meshgrid(new_x_vals, new_y_vals)

        # un ravel the grid
        xx_rav = np.ravel(xx)
        yy_rav = np.ravel(yy)

        # sensitivity will all other feature at their means
        # contstruct a new array
        m = np.median(X, axis=0)
        for i, _ in enumerate(xx_rav):
            if i == 0:
                new_data = m
            else:
                new_data = np.vstack((new_data, m))

        new_data[:, index_x] = xx_rav
        new_data[:, index_y] = yy_rav
        # get points to scatter plot over
        x_new = X[:, index_x]
        y_new = X[:, index_y]

        # predict the new data
        if self.predict_fun:
            new_preds = np.reshape(self.predict_fun(new_data), xx.shape)
        else:
            from xgboost import DMatrix
            new_preds = np.reshape(self.model.predict(DMatrix(new_data, feature_names=self.feature_names)), xx.shape)
        self.new_preds = new_preds
        self.new_features = new_data
        plt.figure()
        plt.contourf(xx, yy, new_preds, cmap=plt.get_cmap('bwr'))
        plt.colorbar()
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title('Decision Boundary')
        plt.show()


def importance_plot(feature_names, imp_array, n=20, path=None, xlab='Information Gain', ylab='Feature',
                    size=(10, 8), fontsize=10):
    '''
     Horriontal Bar Plot Method for Feature Importance

     usage:
     feature_names=['f1', 'f2', 'f3', 'f4']
     imp_array=[.1, .8, .4, 0]
     importance_plot(feature_names, imp_array, n=2)
    :param feature_names: array of feature names
    :param imp_array: array of floating feature importances
    :param n: int number of features to select
    :param path: str file path to save the image , ex: featureImportance.png
    :param xlab: str x label
    :param ylab: str y labled
    :param size: (w, h) tuple of plot size
    :param fontsize: in font size
    :return: barh plot of feature importance
    '''
    n_features = len(feature_names)
    if n_features != len(imp_array):
        raise ValueError('n_features {0} != len imp_array {1}'.format(n_features, len(imp_array)))
    n_selected = min(n_features, n)
    index = np.argsort(imp_array)[::-1][:n_selected][::-1]
    feature_names_selected = np.array(feature_names)[index]
    imp_array_selected = np.array(imp_array)[index]

    fig, ax = plt.subplots(figsize=size)
    pos = list(range(n_selected))
    values = imp_array_selected
    m = np.mean(values)
    names = list(feature_names_selected)
    ax.barh(pos, values, align='center', height=0.5, color='lightblue')
    for i, j in enumerate(pos):
        ax.text(m / 5, j - .1, names[i], fontsize= fontsize)
    ax.set_title('Feature Importance, Top {} Features'.format(n))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yticks([])
    # Save the full figure...
    if isinstance(path, type(None)) is False:
        fig.savefig(path)
    plt.show()
