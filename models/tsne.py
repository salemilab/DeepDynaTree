#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
tsne.py: 
"""

import os
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
plt.rcParams.update({'font.size': 40})

class AlyTSNE(object):
    def __init__(self, feat, perplexity, label, subset, save_folder):
        self.feat = feat
        self.label = label
        self.perplexity = perplexity
        self.subset = subset
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def generate_tsne(self, use_pca=False, overwrite=False, **kwargs):
        """
        Generate the t-sne representation
        :param subset:
        :param use_pca:
        :param kwargs:
        :return:
        """
        pca_dim = kwargs.get("pca_dim", 5)
        random_state = kwargs.get("random_seed", 0)

        if use_pca:
            t_sne_f = osp.join(self.save_folder, "{}_PCA_{}.npy".format(self.subset, pca_dim))
        else:
            t_sne_f = osp.join(self.save_folder, "{}.npy".format(self.subset))
        self.t_sne_f = t_sne_f

        if osp.exists(t_sne_f) and not overwrite:
            print("Not saved! The t-SNE feature file has already exited. {}".format(t_sne_f))
            return 1

        if use_pca:
            pca = PCA(n_components=pca_dim, svd_solver='full', random_state=random_state)
            pca.fit(self.feat)
            reduced_feat = pca.transform(self.feat)
        else:
            reduced_feat = self.feat

        model = TSNE(n_components=2,perplexity=self.perplexity, random_state=random_state)
        x_tsne = model.fit_transform(reduced_feat)
        saved_dict = {"x_tsne": x_tsne, "label": self.label}

        np.save(self.t_sne_f, saved_dict)
        print("{} is done.".format(self.subset))

    def plot(self, **kwargs):
        overwrite = kwargs.get("overwrite", False)
        legend_info = kwargs.get("legend_info", None)

        tsne_plot_path = osp.join(self.save_folder)
        os.makedirs(tsne_plot_path, exist_ok=True)

        tsne_dict = np.load(self.t_sne_f, allow_pickle=True).item()
        tsne_dict["t-SNE-1"] = tsne_dict["x_tsne"][:, 0]
        tsne_dict["t-SNE-2"] = tsne_dict["x_tsne"][:, 1]
        del tsne_dict["x_tsne"]
        plot_df = pd.DataFrame.from_dict(tsne_dict)
        print(plot_df.head())

        plot_fig = osp.join(tsne_plot_path, "{}.png".format(self.subset))
        plot_eps_fig = osp.join(tsne_plot_path, "{}.svg".format(self.subset))

        #sns.set(style="whitegrid")  # whitegrid, darkgrid
        #sns.set_context("poster")

        # With self cmp and markers
        # cmap = sns.dark_palette("#2ecc71", as_cmap=True)
        # cmap = {"wee1-1": "#984ea3", "wee1-0": "#b3cde3",  # 1: #4daf4a; 40: beaed4;
        #         "sahh-1": "#386cb0", "sahh-0": "#fdc086",  # fdc086
        #         "pa2ga-1": "#f0027f", "pa2ga-0": "#ccebc5"}  # ffff99
        # # cmap = sns.color_palette("husl", 6)
        # # cmap = sns.color_palette("Set1", n_colors=8, desat=.5)
        # fig = sns.lmplot(x="t-SNE-1", y="t-SNE-2", data=plot_df, hue="target_label",
        #                  fit_reg=False, legend=False, height=15, aspect=1.6,
        #                  scatter_kws={"alpha": 0.8, "s": 60}, palette=cmap,
        #                  markers=["x", ".", "x", ".", "x", "."])  # palette=cmap

        # Without self cmp
        fig = sns.lmplot(x="t-SNE-1", y="t-SNE-2", data=plot_df, hue="label",
                         fit_reg=False, legend=False, height=15, aspect=1,
                         scatter_kws={"alpha": 0.8, "s":80})  # palette=cmap
        plt.title('Tree shape statistics',fontsize=60)

        # Avg:[height=15 "s": 2000]
        ax = fig.fig.get_axes()[0]

        # Sort the legend order
        # https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
        print(labels)
        #leg = ax.legend(handles, labels, loc="upper right", ncol=1, borderaxespad=0., fontsize=40,bbox_to_anchor=(1.0, 1))
        # leg = ax.legend(loc="best", ncol=1, borderaxespad=0., fontsize=30)  # bbox_to_anchor=(1.05, 1), title="Molecule", title_fontsize=30,
        '''
        if legend_info:
            for t in leg.texts:
                inside_df_legend_label = t.get_text()
                t.set_text(legend_info.get(int(float(inside_df_legend_label)), inside_df_legend_label))
                # print(t.get_text())
        '''
        # ax.tick_params(labelsize=40)  #
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        '''
        for lh in leg.legendHandles:
            lh.set_alpha(0.8)
            lh._sizes = [200]  # Avg:[1000]
        '''
        ax.set_xlabel("", fontsize=60)  #
        ax.set_ylabel("", fontsize=60)  #
        ax.set_xticks([-80,-60,-40,-20,0,20,40,60,80])
        ax.grid(False)

        plt.tight_layout()
        plt.show()
        # return 1
        if os.path.exists(plot_eps_fig) and not overwrite:
            # and os.path.exists(plot_eps_fig)
            print("Not saved! The figure {}.png has already existed.".format(plot_fig))
        else:
            print("Save the plot {}".format(plot_eps_fig))
            fig.savefig(plot_fig, dpi=600)
            fig.savefig(plot_eps_fig, dpi=600)