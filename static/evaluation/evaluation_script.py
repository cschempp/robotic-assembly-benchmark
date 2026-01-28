import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 8
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class AssemblyBenchmark():
    def __init__(self, filepath):
        self.load_csv(filepath)
        self._init_categories()
        self.cm = 1 / 2.54  # centimeters to inches
        self.category_colors = plt.cm.Blues.resampled(len(self.categories))(range(len(self.categories)))

    def _init_categories(self):
        categories = self.dataframe["category"]
        categories = categories.unique()
        self.categories = categories[~pd.isna(categories)]

    def load_csv(self, filepath: str):
        self.dataframe = pd.read_excel(filepath)

    def evaluate(self):
        for idx, row in self.dataframe.iterrows():
            success_ = row[7:]
            repetitions = len(success_)
            complexity_tolerance = row["complexity_tolerance"]
            complexity_geometry = row["complexity_geometry"]
            complexity_material = row["complexity_material"]
            difficulty = complexity_tolerance + complexity_geometry + complexity_material
            reliability = np.sum(success_.values) / repetitions
            score = reliability * difficulty

            self.dataframe.loc[idx, "reliability"] = reliability
            self.dataframe.loc[idx, "difficulty"] = difficulty
            self.dataframe.loc[idx, "score"] = score

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
        self.sorted_data = self.dataframe.sort_values(by="ID")

    def _plot_subtask_reliability(self):
        fig, ax = plt.subplots(figsize=(21 * self.cm, 10 * self.cm))
        ids = self.dataframe["ID"]
        subtask_idx = ~np.isnan(ids)
        ids = ids[subtask_idx]
        reliability = self.dataframe["reliability"][subtask_idx] * 100
        
        ax.bar(ids, reliability, color=self.category_colors[-1], width=0.6)
        ax.set_title("Reliability of each subtask")
        ax.set_xticks(np.arange(min(ids), max(ids) + 1))
        ax.set_xticklabels(ids.to_numpy(dtype=int))
        ax.set_xlabel("Subtask $i$")
        ax.set_ylabel("Reliability $R_i$ [%]")
        plt.tight_layout()
        plt.savefig("01_subtask_reliability.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_category_reliability(self):
        fig, ax = plt.subplots(figsize=(12 * self.cm, 10 * self.cm))
        category_reliability = [
            self.dataframe.loc[self.dataframe["category"] == category, "reliability"] * 100
            for category in self.categories
        ]
        bplot = ax.boxplot(category_reliability, patch_artist=True)
        for patch, median, color in zip(bplot['boxes'], bplot['medians'], self.category_colors):
            patch.set_facecolor(color)
            median.set_color("black")
        ax.set_title("Reliability of each category")
        ax.set_xlabel("Category")
        ax.set_xticks(range(1, len(self.categories) + 1))
        ax.set_xticklabels(self.categories)
        ax.set_ylabel("Reliability $R_c$ [%]")
        plt.tight_layout()
        plt.savefig("02_category_reliability.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_subtask_scores(self):
        fig, ax = plt.subplots(figsize=(22 * self.cm, 10 * self.cm))
        ids = self.dataframe["ID"]
        subtask_idx = ~np.isnan(ids)
        ids = ids[subtask_idx]
        scores = self.dataframe["score"][subtask_idx]
        
        ax.bar(ids, scores, color=self.category_colors[-1], width=0.6)
        ax.set_title("Score of each subtask")
        ax.set_xticks(np.arange(min(ids), max(ids) + 1))
        ax.set_xticklabels(ids.to_numpy(dtype=int))
        ax.set_xlabel("Subtask $i$")
        ax.set_ylabel("Score $S_i$")
        plt.tight_layout()
        plt.savefig("01b_subtask_scores.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_score_by_category(self):
        fig, ax = plt.subplots(figsize=(11 * self.cm, 10 * self.cm))
        category_scores = [
            self.dataframe.loc[self.dataframe["category"] == category, "score"].sum()
            for category in self.categories
        ]
        category_difficulties = [
            self.dataframe.loc[self.dataframe["category"] == category, "difficulty"].sum()
            for category in self.categories
        ]
        bottom_scores, bottom_difficulties = 0, 0
        for score, max_score, color, category in zip(category_scores, category_difficulties, self.category_colors, self.categories):
            ax.bar(["Total Score"], [score], bottom=bottom_scores, color=color, edgecolor="black", label=category)
            bottom_scores += score
            ax.bar(["Max Score"], [max_score], bottom=bottom_difficulties, color=color, edgecolor="black")
            bottom_difficulties += max_score
        ax.set_title("Total score contribution by category")
        ax.set_ylabel("Score $S_c$")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("03_score_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_score_by_subassembly(self):
        fig, ax = plt.subplots(figsize=(11 * self.cm, 10 * self.cm))
        subassembly_indices = self.dataframe[self.dataframe["process"].str.contains("subassembly")].index
        subassembly_scores = [
            self.dataframe.loc[:sub_idx - 1, "score"].sum() if i == 0 else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "score"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        subassembly_difficulties = [
            self.dataframe.loc[:sub_idx - 1, "difficulty"].sum() if i == 0 else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "difficulty"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        bottom_scores, bottom_maxscore = 0, 0

        subassembly_colors = plt.cm.Blues.resampled(len(subassembly_indices))(range(len(subassembly_indices)))

        for score, color, max_score, subassembly in zip(subassembly_scores, subassembly_colors, subassembly_difficulties, [f"SA{i+1}" for i in range(len(subassembly_indices))]):
            ax.bar(["Total Score"], [score], bottom=bottom_scores, color=color, edgecolor="black", label=subassembly)
            bottom_scores += score
            ax.bar(["Max Score"], [max_score], bottom=bottom_maxscore, color=color, edgecolor="black")
            bottom_maxscore += max_score
        ax.set_title("Total score contribution by subassembly")
        ax.set_ylabel("Score $S_{sa}$")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("04_score_by_subassembly.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_table(self):
        subassembly_indices = self.dataframe[self.dataframe["process"].str.contains("subassembly")].index
        subassembly_scores = [
            self.dataframe.loc[:sub_idx - 1, "score"].sum() if i == 0 else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "score"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        subassembly_difficulties = [
            self.dataframe.loc[:sub_idx - 1, "difficulty"].sum() if i == 0 else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "difficulty"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        
        category_scores = [
            self.dataframe.loc[self.dataframe["category"] == category, "score"].sum()
            for category in self.categories
        ]
        category_difficulties = [
            self.dataframe.loc[self.dataframe["category"] == category, "difficulty"].sum()
            for category in self.categories
        ]

        table_data = []
        for subassembly, score, max_score in zip([f"SA{i+1}" for i in range(len(subassembly_indices))], subassembly_scores, subassembly_difficulties):
            difference = max_score - score
            rel = score / max_score if max_score > 0 else 0
            table_data.append([subassembly, f"{score:.2f}", f"{max_score:.2f}", f"{difference:.2f}", f"{rel:.2%}"])

        for category, score, max_score in zip(self.categories, category_scores, category_difficulties):
            difference = max_score - score
            rel = score / max_score if max_score > 0 else 0
            table_data.append([category, f"{score:.2f}", f"{max_score:.2f}", f"{difference:.2f}", f"{rel:.2%}"])

        return table_data

    def _plot_summary_table(self):
        fig, ax = plt.subplots(figsize=(15 * self.cm, 10 * self.cm))
        ax.axis("tight")
        ax.axis("off")

        table_data = self._create_summary_table()
        col_labels = ["SA/Category", "Score", "Max. Score", "Difference", "Rel. Score"]
        
        table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col_labels)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")

        differences = [float(row[-2]) for row in table_data]
        min_diff_idx = differences.index(min(differences)) + 1
        max_diff_idx = differences.index(max(differences)) + 1

        for (row, col), cell in table.get_celld().items():
            if row == min_diff_idx:
                cell.set_facecolor("#d9ead3")
            elif row == max_diff_idx:
                cell.set_facecolor("#f4cccc")

        plt.tight_layout()
        plt.savefig("05_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_detailed_table(self):
        table_data = []
        subassembly_ranges = []
        current_subassembly = None
        start_idx = None

        for idx, row in self.dataframe.iterrows():
            reliability = f"{row['reliability']:.2f}"
            difficulty = int(row["difficulty"])
            score = f"{row['score']:.2f}"

            if "assembly" in row["process"]:
                table_data.append([row["process"], "", reliability, difficulty, score])
            elif "subassembly" in row["process"]:
                if current_subassembly is not None:
                    subassembly_ranges.append((current_subassembly, start_idx, len(table_data) - 1))
                current_subassembly = row["process"]
                start_idx = len(table_data)
                table_data.append([row["process"], "", reliability, difficulty, score])
            else:
                table_data.append(["", int(row["ID"]), reliability, difficulty, score])

        if current_subassembly is not None:
            subassembly_ranges.append((current_subassembly, start_idx, len(table_data) - 1))

        return table_data, subassembly_ranges

    def _plot_detailed_table(self):
        fig, ax = plt.subplots(figsize=(15 * self.cm, 29.7 * self.cm))
        ax.axis("tight")
        ax.axis("off")

        table_data, subassembly_ranges = self._create_detailed_table()
        col_labels = ["Subassembly", "Subtask ID", "Reliability $R$", "Points $P$", "Score $S$"]
        
        table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col_labels)

        for subassembly, start, end in subassembly_ranges:
            for row in range(start + 1, end + 1):
                table._cells[(row + 1, 0)].visible = False
            cell = table._cells[(start + 1, 0)]
            cell.set_text_props(text=subassembly, ha="center", va="center", weight="bold")
            cell.set_height((end - start + 1) * cell.get_height())

        for key in table._cells:
            cell = table._cells[key]
            if key[0] > 0:
                if "subassembly" in table_data[key[0] - 1][0]:
                    cell.set_facecolor("#d9ead3")
                elif "assembly" in table_data[key[0] - 1][0]:
                    cell.set_facecolor("#c9daf8")
                else:
                    cell.set_facecolor("white")

        plt.tight_layout()
        plt.savefig("06_detailed_table.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_evaluation_protocol(self):
        """Generate individual PNG plots and combined PDF protocol"""
        self._plot_subtask_reliability()
        self._plot_category_reliability()

        self._plot_subtask_scores()
        self._plot_score_by_category()
        self._plot_score_by_subassembly()

        self._plot_summary_table()
        self._plot_detailed_table()

        # Create combined PDF
        pdf_filename = "evaluation_protocol.pdf"
        with PdfPages(pdf_filename) as pdf:
            figs = [
                "01_subtask_reliability.png",
                "02_category_reliability.png",
                "03_score_by_category.png",
                "04_score_by_subassembly.png",
                "05_summary_table.png",
                "06_detailed_table.png"
            ]
            for fig_file in figs:
                if os.path.exists(fig_file):
                    img = plt.imread(fig_file)
                    fig, ax = plt.subplots(figsize=(21 * self.cm, 29.7 * self.cm))
                    ax.imshow(img)
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

        print(f"Evaluation protocol saved as {pdf_filename}")
        print("Individual plots saved as 01_*.png through 06_*.png")

    def _plot_combined_scores(self):
        """Plot subtask scores, category scores, and subassembly scores in one figure"""
        fig = plt.figure(figsize=(21 * self.cm, 5 * self.cm))
        gs = fig.add_gridspec(1, 3, width_ratios=[6, 2, 2])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        
        # Plot 1: Subtask scores
        ids = self.dataframe["ID"]
        subtask_idx = ~np.isnan(ids)
        ids = ids[subtask_idx]
        scores = self.dataframe["score"][subtask_idx]
        difficulties = self.dataframe["difficulty"][subtask_idx]
        
        axes[0].bar(ids, difficulties, color="lightblue", width=0.6, label="max. score")
        axes[0].bar(ids, scores, color=self.category_colors[-1], width=0.6, label="achieved score")
        axes[0].set_xticks(np.arange(min(ids), max(ids) + 1)[::2])
        axes[0].set_xticklabels(ids.to_numpy(dtype=int)[::2])
        axes[0].set_xlabel("Subtask $i$")
        axes[0].set_ylabel("Score $S_i$")
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.2), ncol=2, frameon=False)
        
        # Plot 2: Score by category (bar per category)
        category_scores = [
            self.dataframe.loc[self.dataframe["category"] == category, "score"].sum()
            for category in self.categories
        ]
        category_difficulties = [
            self.dataframe.loc[self.dataframe["category"] == category, "difficulty"].sum()
            for category in self.categories
        ]
        x_pos = np.arange(len(self.categories))
        width = 0.35
        axes[1].bar(x_pos, category_difficulties, width=0.6, color="lightblue", edgecolor="black", label="Max Score")
        axes[1].bar(x_pos, category_scores, width=0.6, color=self.category_colors[-1], edgecolor="black", label="Score")
        axes[1].set_xlabel("Category")
        axes[1].set_ylabel("Score $S_c$")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(self.categories, rotation=0)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Score by subassembly (bar per subassembly)
        subassembly_indices = self.dataframe[self.dataframe["process"].str.contains("subassembly")].index
        subassembly_scores = [
            self.dataframe.loc[:sub_idx - 1, "score"].sum() if i == 0 else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "score"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        subassembly_difficulties = [
            self.dataframe.loc[:sub_idx - 1, "difficulty"].sum() if i == 0 else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "difficulty"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        x_pos = np.arange(len(subassembly_indices))
        subassembly_labels = [f"SA{i+1}" for i in range(len(subassembly_indices))]
        axes[2].bar(x_pos, subassembly_difficulties, width=0.6, color="lightblue", edgecolor="black")
        axes[2].bar(x_pos, subassembly_scores, width=0.6, color=self.category_colors[-1], edgecolor="black")
        axes[2].set_xlabel("Subassembly")
        axes[2].set_ylabel("Score $S_{sa}$")
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(subassembly_labels)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("07_combined_scores.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_combined_reliability(self):
        """Plot subtask reliability, category reliability, and subassembly reliability in one figure"""
        fig = plt.figure(figsize=(21 * self.cm, 4.5 * self.cm))
        gs = fig.add_gridspec(1, 3, width_ratios=[6, 2, 2])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        
        # Plot 1: Subtask reliability
        ids = self.dataframe["ID"]
        subtask_idx = ~np.isnan(ids)
        ids = ids[subtask_idx]
        reliability = self.dataframe["reliability"][subtask_idx] * 100
        axes[0].bar(ids, reliability, color=self.category_colors[-1], width=0.6)
        axes[0].set_xticks(np.arange(min(ids), max(ids) + 1)[::2])
        axes[0].set_xticklabels(ids.to_numpy(dtype=int)[::2])
        axes[0].set_xlabel("Subtask $i$")
        axes[0].set_ylabel("Reliability $R_i$ [%]")
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 100])
        
        # Plot 2: Category reliability (boxplot)
        category_reliability = [
            self.dataframe.loc[self.dataframe["category"] == category, "reliability"] * 100
            for category in self.categories
        ]
        bplot = axes[1].boxplot(category_reliability, patch_artist=True)
        for patch, median in zip(bplot['boxes'], bplot['medians']):
            patch.set_facecolor(self.category_colors[-1])
            median.set_color("black")
        axes[1].set_xlabel("Category")
        axes[1].set_xticks(range(1, len(self.categories) + 1))
        axes[1].set_xticklabels(self.categories)
        axes[1].set_ylabel("Reliability $R_c$ [%]")
        axes[1].set_ylim([0, 100])
        
        # Plot 3: Subassembly reliability
        subassembly_indices = self.dataframe[self.dataframe["process"].str.contains("subassembly")].index
        subassembly_reliability = [
            self.dataframe.loc[subassembly_indices[i], "reliability"] * 100
            for i in range(len(subassembly_indices))
        ]
        axes[2].bar([f"SA{i+1}" for i in range(len(subassembly_indices))], subassembly_reliability, color=self.category_colors[-1], edgecolor="black", width=0.6)
        axes[2].set_xlabel("Subassembly")
        axes[2].set_ylabel("Reliability $R_{sa}$ [%]")
        axes[2].set_ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig("08_combined_reliability.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def visualize_results(self):
        self.generate_evaluation_protocol()
        self._plot_combined_scores()
        self._plot_combined_reliability()

if __name__ == "__main__":
    asm = AssemblyBenchmark(filepath=r"evaluation\benchmark_protocol_sheet.xlsx")
    asm.evaluate()
    asm.visualize_results()
