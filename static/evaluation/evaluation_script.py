import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 8


class AssemblyBenchmark():
    def __init__(self, filepath):
        self.load_csv(filepath)

        categories = self.dataframe["category"]
        categories = categories.unique()
        self.categories = categories[~pd.isna(categories)]

        cmap = mpl.colormaps['Blues']
        self.colors = cmap(np.linspace(0.2, 1, len(self.categories)))
    
    def load_csv(self, filepath: str):
        self.dataframe = pd.read_excel(filepath)
    
    def evaluate(self):

        for idx, row in self.dataframe.iterrows():
            success_ = row[5:]
            repetitions = len(success_)

            complexity_tolerance = row["complexity_tolerance"]
            complexity_geometry = row["complexity_geometry"]
            complexity_material = row["complexity_material"]

            difficulty = complexity_tolerance + complexity_geometry + complexity_material
            reliability = np.sum(success_.values)/repetitions
            score = reliability * difficulty

            self.dataframe.loc[idx, "reliability"] = reliability
            self.dataframe.loc[idx, "difficulty"] = difficulty
            self.dataframe.loc[idx, "score"] = score

        # total subassembly reliability, score and difficulty
        indices = np.squeeze(np.where(["subassembly" in process for process in self.dataframe["process"]]))
        for i in range(len(indices)):
            if i == 0:
                self.dataframe.loc[indices[i], "reliability"] = np.prod(self.dataframe["reliability"][0:indices[i]])
                self.dataframe.loc[indices[i], "score"] = np.sum(self.dataframe["score"][0:indices[i]])
                self.dataframe.loc[indices[i], "difficulty"] = np.sum(self.dataframe["difficulty"][0:indices[i]])
            else:
                self.dataframe.loc[indices[i], "reliability"] = np.prod(self.dataframe["reliability"][indices[i-1]+1:indices[i]])
                self.dataframe.loc[indices[i], "score"] = np.sum(self.dataframe["score"][indices[i-1]+1:indices[i]])
                self.dataframe.loc[indices[i], "difficulty"] = np.sum(self.dataframe["difficulty"][indices[i-1]+1:indices[i]])
            

        self.idx_asm = int(np.squeeze(np.where(self.dataframe["process"] == "assembly")))

        rels = self.dataframe.loc[indices, "reliability"]
        self.dataframe.loc[self.idx_asm, "reliability"] = np.prod(rels.to_numpy())

        scores = self.dataframe.loc[indices, "score"]
        self.dataframe.loc[self.idx_asm, "score"] = np.sum(scores.to_numpy())

        diffs = self.dataframe.loc[indices, "difficulty"]
        self.dataframe.loc[self.idx_asm, "difficulty"] = np.sum(diffs.to_numpy())
               
    def visualize_results(self):
        cm = 1/2.54 # centimeters in inches
        fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(10*cm, 4*cm), gridspec_kw={'width_ratios': [3, 1]})
        self.plot_categories_reliability_boxplots(axs)
        self.plot_categories_scores_bars(axs)
        plt.subplots_adjust(wspace=0.5)
        
        plt.savefig("benchmark_evaluation.pdf", bbox_inches="tight")
    
    def plot_categories_reliability_boxplots(self,axs):
        # boxplots for each category reliability
        data_to_plot = []
        for category in self.categories:
            category_ = self.dataframe[self.dataframe["category"] == category]
            data_to_plot.append(category_["reliability"]*100)

        bplot = axs[0,0].boxplot(x=data_to_plot,
                    autorange=True,
                    patch_artist=True
                    )
        for patch,median,c in zip(bplot['boxes'], bplot['medians'], self.colors):
            patch.set_facecolor(c)
            median.set_color("black")

        axs[0,0].set_ylabel("reliability [%]")
        axs[0,0].set_xticks([])

    def plot_categories_scores_bars(self,axs):
        width=0.2
        maxpoints = self.dataframe.loc[self.idx_asm, "difficulty"]
        data_to_plot_score = []
        data_to_plot_difficulty = []
        legend = []
        for i, category in enumerate(self.categories):
            category_ = self.dataframe[self.dataframe["category"] == category]
            data_to_plot_score.append(np.sum(category_["score"])/maxpoints*100)
            data_to_plot_difficulty.append(np.sum(category_["difficulty"])/maxpoints*100)
            legend.append(category)

            axs[0, 1].bar([2*width+width/4,width], 
                        [data_to_plot_difficulty[i],data_to_plot_score[i]], 
                        width, 
                        label=category, 
                        tick_label= ["max.", "baseline"], 
                        bottom=[np.sum(data_to_plot_difficulty[:i]),np.sum(data_to_plot_score[:i])], 
                        color=self.colors[i])

        axs[0,1].set_ylabel("score [%]")
        plt.legend(loc='upper center', bbox_to_anchor=(-1.75, -0.2), ncols=len(self.categories), alignment="left")


if __name__ == "__main__":
    asm = AssemblyBenchmark(filepath = r"benchmark_protocol.xlsx")
    asm.evaluate()
    asm.visualize_results()
