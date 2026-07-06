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
        # self.category_colors = plt.cm.Pastel2.colors[:len(self.categories)]
        self.category_colors = plt.cm.Blues.resampled(len(self.categories))(range(len(self.categories)))

    def _init_categories(self):
        categories = self.dataframe["category"]
        categories = categories.unique()
        self.categories = categories[~pd.isna(categories)]

    def load_csv(self, filepath: str):
        self.dataframe = pd.read_excel(filepath)
        # Merged "subtask" cells only carry a value in their top row; fill downward
        # within primitive rows (rows that have a numeric ID) so each primitive knows its subtask.
        primitive_mask = self.dataframe["ID"].notna()
        self.dataframe.loc[primitive_mask, "subtask"] = (
            self.dataframe.loc[primitive_mask, "subtask"].ffill()
        )

    def evaluate(self):
        trial_cols = [c for c in self.dataframe.columns if isinstance(c, int)]

        for idx, row in self.dataframe.iterrows():
            success_ = row[trial_cols]
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

        indices = np.squeeze(np.where(["subassembly" in primitive for primitive in self.dataframe["primitive"]]))
        for i in range(len(indices)):
            if i == 0:
                self.dataframe.loc[indices[i], "reliability"] = np.prod(self.dataframe["reliability"][0:indices[i]])
                self.dataframe.loc[indices[i], "score"] = np.sum(self.dataframe["score"][0:indices[i]])
                self.dataframe.loc[indices[i], "difficulty"] = np.sum(self.dataframe["difficulty"][0:indices[i]])
            else:
                self.dataframe.loc[indices[i], "reliability"] = np.prod(self.dataframe["reliability"][indices[i-1]+1:indices[i]])
                self.dataframe.loc[indices[i], "score"] = np.sum(self.dataframe["score"][indices[i-1]+1:indices[i]])
                self.dataframe.loc[indices[i], "difficulty"] = np.sum(self.dataframe["difficulty"][indices[i-1]+1:indices[i]])

        self.idx_asm = int(np.squeeze(np.where(self.dataframe["primitive"] == "assembly")))
        rels = self.dataframe.loc[indices, "reliability"]
        self.dataframe.loc[self.idx_asm, "reliability"] = np.prod(rels.to_numpy())
        scores = self.dataframe.loc[indices, "score"]
        self.dataframe.loc[self.idx_asm, "score"] = np.sum(scores.to_numpy())
        diffs = self.dataframe.loc[indices, "difficulty"]
        self.dataframe.loc[self.idx_asm, "difficulty"] = np.sum(diffs.to_numpy())
        self.sorted_data = self.dataframe.sort_values(by="ID")

        # Subtask reliability: a trial counts only when ALL primitives of the subtask succeeded.
        # Subtask score: sum of primitive scores; max score: sum of primitive difficulties.
        primitive_rows = self.dataframe[self.dataframe["ID"].notna()]
        self.subtask_reliability = {}
        self.subtask_scores = {}
        self.subtask_max_scores = {}
        for subtask_name, group in primitive_rows.groupby("subtask", sort=False):
            per_trial_success = (group[trial_cols] == 1).all(axis=0)
            self.subtask_reliability[subtask_name] = float(per_trial_success.mean())
            self.subtask_scores[subtask_name] = float(group["score"].sum())
            self.subtask_max_scores[subtask_name] = float(group["difficulty"].sum())

    def _plot_primitive_reliability(self):
        fig, ax = plt.subplots(figsize=(21 * self.cm, 10 * self.cm))
        ids = self.dataframe["ID"]
        subtask_idx = ~np.isnan(ids)
        ids = ids[subtask_idx]
        reliability = self.dataframe["reliability"][subtask_idx] * 100
        
        ax.bar(ids, reliability, color=self.category_colors[-1], width=0.6)
        ax.set_title("Reliability of each primitive")
        ax.set_xticks(np.arange(min(ids), max(ids) + 1))
        ax.set_xticklabels(ids.to_numpy(dtype=int))
        ax.set_xlabel("Primitive $i$")
        ax.set_ylabel("Reliability $R_i$ [%]")
        plt.tight_layout()
        plt.savefig("01_primitive_reliability.png", dpi=300, bbox_inches='tight')
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

    def _plot_primitive_scores(self):
        fig, ax = plt.subplots(figsize=(22 * self.cm, 10 * self.cm))
        ids = self.dataframe["ID"]
        subtask_idx = ~np.isnan(ids)
        ids = ids[subtask_idx]
        scores = self.dataframe["score"][subtask_idx]
        
        ax.bar(ids, scores, color=self.category_colors[-1], width=0.6)
        ax.set_title("Score of each primitive")
        ax.set_xticks(np.arange(min(ids), max(ids) + 1))
        ax.set_xticklabels(ids.to_numpy(dtype=int))
        ax.set_xlabel("Primitive $i$")
        ax.set_ylabel("Score $S_i$")
        plt.tight_layout()
        plt.savefig("01b_primitive_scores.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_score_by_category(self):
        handling_categories = [c for c in self.categories if c.lower() in ("grasp", "reorient")]
        joining_categories = [c for c in self.categories if c.lower() not in ("grasp", "reorient")]

        handling_colors = plt.cm.Blues.resampled(len(handling_categories))(range(len(handling_categories))) if handling_categories else []
        joining_colors = plt.cm.Blues.resampled(len(joining_categories))(range(len(joining_categories))) if joining_categories else []

        fig, axes = plt.subplots(1, 2, figsize=(18 * self.cm, 10 * self.cm))

        for ax, group_categories, group_colors, title in [
            (axes[0], handling_categories, handling_colors, "Handling categories"),
            (axes[1], joining_categories, joining_colors, "Joining categories"),
        ]:
            group_scores = [
                self.dataframe.loc[self.dataframe["category"] == cat, "score"].sum()
                for cat in group_categories
            ]
            group_difficulties = [
                self.dataframe.loc[self.dataframe["category"] == cat, "difficulty"].sum()
                for cat in group_categories
            ]
            bottom_scores, bottom_difficulties = 0, 0
            for score, max_score, color, category in zip(group_scores, group_difficulties, group_colors, group_categories):
                ax.bar(["Total Score"], [score], bottom=bottom_scores, color=color, edgecolor="black", label=category)
                bottom_scores += score
                ax.bar(["Max Score"], [max_score], bottom=bottom_difficulties, color=color, edgecolor="black")
                bottom_difficulties += max_score
            ax.set_title(title)
            ax.set_ylabel("Score $S_c$")
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig("03_score_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_score_by_subassembly(self):
        fig, ax = plt.subplots(figsize=(11 * self.cm, 10 * self.cm))
        subassembly_indices = self.dataframe[self.dataframe["primitive"].str.contains("subassembly")].index
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
        subassembly_indices = self.dataframe[self.dataframe["primitive"].str.contains("subassembly")].index
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

            if "assembly" in row["primitive"]:
                table_data.append([row["primitive"], "", reliability, difficulty, score])
            elif "subassembly" in row["primitive"]:
                if current_subassembly is not None:
                    subassembly_ranges.append((current_subassembly, start_idx, len(table_data) - 1))
                current_subassembly = row["primitive"]
                start_idx = len(table_data)
                table_data.append([row["primitive"], "", reliability, difficulty, score])
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

    def _plot_subtask_reliability(self):
        fig, ax = plt.subplots(figsize=(21 * self.cm, 10 * self.cm))
        subtask_names = list(self.subtask_reliability.keys())
        reliabilities = [self.subtask_reliability[s] * 100 for s in subtask_names]
        x_pos = np.arange(len(subtask_names))
        ax.bar(x_pos, reliabilities, color=self.category_colors[-1], width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subtask_names, rotation=45, ha='right')
        ax.set_xlabel("Subtask")
        ax.set_ylabel("Reliability $R_{st}$ [%]")
        ax.set_ylim([0, 100])
        ax.set_title("Reliability of each subtask")
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("09_subtask_reliability.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_subtask_score(self):
        fig, ax = plt.subplots(figsize=(21 * self.cm, 10 * self.cm))
        subtask_names = list(self.subtask_scores.keys())
        scores = [self.subtask_scores[s] for s in subtask_names]
        max_scores = [self.subtask_max_scores[s] for s in subtask_names]
        x_pos = np.arange(len(subtask_names))
        ax.bar(x_pos, max_scores, color="lightblue", width=0.6, label="max. score")
        ax.bar(x_pos, scores, color=self.category_colors[-1], width=0.6, label="achieved score")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subtask_names, rotation=45, ha='right')
        ax.set_xlabel("Subtask")
        ax.set_ylabel("Score $S_{st}$")
        ax.set_title("Score of each subtask")
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("10_subtask_score.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _export_latex_table(self, max_data_cols: int = 10):
        """Transposed layout matching the reference image.

        Columns are primitives grouped by subtask; rows are attributes
        (ID, category, R, score).  Each subtask gets one extra "total" column
        on its right.  Subtasks are packed into blocks of at most
        *max_data_cols* data columns; when a block is full a new block starts
        below (same SA).  Each subassembly gets a bold label line before its
        blocks.

        Required LaTeX packages: booktabs, tabular* (standard), array.
        """
        def esc(s):
            return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

        def fmt_r(r_pct):
            return f"{r_pct:.0f}"

        def fmt_score(score, max_score):
            s = f"{score:g}"
            return f"{s} / {int(max_score)}"

        # Build ordered structure: [(sa_name, [(subtask_name, [prim_rows])])]
        # SA summary rows sit AFTER their primitives in the dataframe; when we
        # encounter the SA row we know the name to assign to the buffered group.
        structure = []
        buf: list = []          # subtasks accumulated for the current SA
        current_subtask: str | None = None

        for _, row in self.dataframe.iterrows():
            prim = row["primitive"]
            if prim == "assembly":
                continue
            elif "subassembly" in prim:
                structure.append((prim, buf))
                buf = []
                current_subtask = None
            else:
                subtask = row["subtask"]
                if subtask != current_subtask:
                    current_subtask = subtask
                    buf.append((subtask, []))
                buf[-1][1].append(row)

        N = 1 + max_data_cols  # label col + data cols
        col_spec = f"p{{2.3cm}}@{{\\extracolsep{{\\fill}}}}" + "c" * max_data_cols

        lines = []
        lines.append(r"% Required packages: booktabs, array")
        lines.append(r"% Wrap in {\footnotesize ...} or a table float as needed.")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(r"\renewcommand{\arraystretch}{0.85}")
        lines.append(f"\\begin{{tabular*}}{{\\linewidth}}{{{col_spec}}}")
        lines.append(r"\toprule")

        for sa_idx, (sa_name, subtasks) in enumerate(structure):
            # Split subtasks into blocks; each subtask occupies (n_prims + 1) columns.
            blocks: list[list] = []
            cur_block: list = []
            cur_cols = 0
            for st_name, prims in subtasks:
                n = len(prims) + 1
                if cur_block and cur_cols + n > max_data_cols:
                    blocks.append(cur_block)
                    cur_block = []
                    cur_cols = 0
                cur_block.append((st_name, prims))
                cur_cols += n
            if cur_block:
                blocks.append(cur_block)

            # SA separator: midrule before every SA except the first
            if sa_idx > 0:
                lines.append(r"\midrule")

            # SA header row spanning all columns
            lines.append(
                f"\\multicolumn{{{N}}}{{l}}{{\\textbf{{{esc(sa_name)}}}}}" + r" \\"
            )
            lines.append(r"\midrule")

            for block_idx, block in enumerate(blocks):
                if block_idx > 0:
                    lines.append(r"\addlinespace[4pt]")

                block_cols = sum(len(prims) + 1 for _, prims in block)
                pad = max_data_cols - block_cols

                # ── Row 1: subtask names + "total" headers ─────────────────────
                r1 = [r"\textit{subtask}"]
                for st_name, prims in block:
                    n_p = len(prims)
                    r1.append(f"\\multicolumn{{{n_p}}}{{c}}{{{esc(st_name)}}}")
                    r1.append("total")
                if pad > 0:
                    r1.append(f"\\multicolumn{{{pad}}}{{c}}{{}}")
                lines.append(" & ".join(r1) + r" \\")

                # Partial rules under each subtask's primitive columns
                col_idx = 2
                cmids = []
                for _, prims in block:
                    n_p = len(prims)
                    cmids.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_p - 1}}}")
                    col_idx += n_p + 1
                lines.append("".join(cmids))

                # ── Row 2: primitive IDs ───────────────────────────────────────
                r2 = ["ID"]
                for _, prims in block:
                    for prow in prims:
                        r2.append(str(int(prow["ID"])))
                    r2.append("")
                r2.extend([""] * pad)
                lines.append(" & ".join(r2) + r" \\")

                # ── Row 3: categories ──────────────────────────────────────────
                r3 = ["Cat."]
                for _, prims in block:
                    for prow in prims:
                        cat = esc(prow["category"]) if pd.notna(prow["category"]) else ""
                        r3.append(cat)
                    r3.append("")
                r3.extend([""] * pad)
                lines.append(" & ".join(r3) + r" \\")

                lines.append(r"\midrule")

                # ── Row 4: reliability ─────────────────────────────────────────
                r4 = [r"$R$\,[\%]"]
                for st_name, prims in block:
                    for prow in prims:
                        r4.append(fmt_r(prow["reliability"] * 100))
                    st_r = self.subtask_reliability.get(st_name, 0) * 100
                    r4.append(fmt_r(st_r))
                r4.extend([""] * pad)
                lines.append(" & ".join(r4) + r" \\")

                # ── Row 5: score (achieved / max) ──────────────────────────────
                r5 = ["Score"]
                for st_name, prims in block:
                    for prow in prims:
                        r5.append(fmt_score(prow["score"], prow["difficulty"]))
                    st_score = self.subtask_scores.get(st_name, 0)
                    st_max = self.subtask_max_scores.get(st_name, 0)
                    r5.append(fmt_score(st_score, st_max))
                r5.extend([""] * pad)
                lines.append(" & ".join(r5) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")

        with open("11_score_reliability_table.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("LaTeX table saved as 11_score_reliability_table.tex")

    def _export_latex_summary_tables(self):
        """Export compact summary tables for categories, subtasks, and subassemblies.

        Writes 12_summary_tables.tex with three tabular* environments:
          1. Category table  – per-category R stats and score
          2. Subtask table   – per-subtask R and score, grouped by SA
          3. Subassembly table – per-SA R and score
        Required packages: booktabs, array.
        """
        def esc(s):
            return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

        def fmt_r(r_frac):
            return f"{r_frac * 100:.0f}"

        def fmt_score(score, max_score):
            s = f"{score:g}"
            return f"{s} / {int(max_score)}"

        lines = []
        lines.append(r"% Required packages: booktabs, array")
        lines.append(r"% Wrap blocks in {\footnotesize ...} or table floats as needed.")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(r"\renewcommand{\arraystretch}{0.9}")

        # ── Table 1: Category ────────────────────────────────────────────────
        lines.append("")
        lines.append(r"\noindent\textbf{Scores and reliability by category}\par\noindent")
        lines.append(r"\begin{tabular*}{\linewidth}{p{2.5cm}@{\extracolsep{\fill}}ccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"Category & $N$ & $R_{\min}$\,[\%] & $\bar{R}$\,[\%] & "
            r"$R_{\max}$\,[\%] & Score \\"
        )
        lines.append(r"\midrule")
        primitive_rows = self.dataframe[self.dataframe["ID"].notna()]
        for cat in self.categories:
            cat_rows = primitive_rows[primitive_rows["category"] == cat]
            n = len(cat_rows)
            rels = cat_rows["reliability"]
            r_min = fmt_r(rels.min())
            r_mean = fmt_r(rels.mean())
            r_max = fmt_r(rels.max())
            score = cat_rows["score"].sum()
            max_score = cat_rows["difficulty"].sum()
            lines.append(
                f"{esc(cat)} & {n} & {r_min} & {r_mean} & {r_max} & "
                f"{fmt_score(score, max_score)} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")

        # ── Table 2: Subtask (grouped by SA) ────────────────────────────────
        lines.append("")
        lines.append(r"\smallskip")
        lines.append(r"\noindent\textbf{Scores and reliability by subtask}\par\noindent")
        lines.append(r"\begin{tabular*}{\linewidth}{p{4cm}@{\extracolsep{\fill}}cc}")
        lines.append(r"\toprule")
        lines.append(r"Subtask & $R$\,[\%] & Score \\")
        lines.append(r"\midrule")

        # Reuse the same SA-buffer pattern as _export_latex_table
        structure = []
        buf: list = []
        current_subtask: str | None = None
        for _, row in self.dataframe.iterrows():
            prim = row["primitive"]
            if prim == "assembly":
                continue
            elif "subassembly" in prim:
                structure.append((prim, buf))
                buf = []
                current_subtask = None
            else:
                subtask = row["subtask"]
                if subtask != current_subtask:
                    current_subtask = subtask
                    buf.append(subtask)

        first_sa = True
        for sa_name, subtask_names in structure:
            if not first_sa:
                lines.append(r"\midrule")
            first_sa = False
            lines.append(
                f"\\multicolumn{{3}}{{l}}{{\\textbf{{{esc(sa_name)}}}}}" + r" \\"
            )
            lines.append(r"\midrule")
            for st_name in subtask_names:
                r_val = fmt_r(self.subtask_reliability.get(st_name, 0))
                score = self.subtask_scores.get(st_name, 0)
                max_score = self.subtask_max_scores.get(st_name, 0)
                lines.append(f"{esc(st_name)} & {r_val} & {fmt_score(score, max_score)} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")

        # ── Table 3: Subassembly ─────────────────────────────────────────────
        lines.append("")
        lines.append(r"\smallskip")
        lines.append(r"\noindent\textbf{Scores and reliability by subassembly}\par\noindent")
        lines.append(r"\begin{tabular*}{\linewidth}{p{3cm}@{\extracolsep{\fill}}cc}")
        lines.append(r"\toprule")
        lines.append(r"Subassembly & $R$\,[\%] & Score \\")
        lines.append(r"\midrule")
        sa_rows = self.dataframe[self.dataframe["primitive"].str.contains("subassembly")]
        for i, (_, row) in enumerate(sa_rows.iterrows()):
            sa_label = f"SA{i + 1}"
            r_val = fmt_r(row["reliability"])
            score = row["score"]
            max_score = row["difficulty"]
            lines.append(f"{sa_label} & {r_val} & {fmt_score(score, max_score)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")

        with open("12_summary_tables.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Summary tables saved as 12_summary_tables.tex")

    def generate_evaluation_protocol(self):
        """Generate individual PNG plots and combined PDF protocol"""
        self._plot_primitive_reliability()
        self._plot_category_reliability()

        self._plot_primitive_scores()
        self._plot_score_by_category()
        self._plot_score_by_subassembly()

        self._plot_summary_table()
        self._plot_detailed_table()
        self._plot_subtask_reliability()
        self._plot_subtask_score()
        self._export_latex_table()
        self._export_latex_summary_tables()

        # Create combined PDF
        pdf_filename = "evaluation_protocol.pdf"
        with PdfPages(pdf_filename) as pdf:
            figs = [
                "01_primitive_reliability.png",
                "02_category_reliability.png",
                "03_score_by_category.png",
                "04_score_by_subassembly.png",
                "05_summary_table.png",
                "06_detailed_table.png",
                "09_subtask_reliability.png",
                "10_subtask_score.png",
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
        """2×2 grid: primitive scores (top-left), category scores (top-right),
        subtask scores (bottom-left), subassembly scores (bottom-right)."""
        fig, axes = plt.subplots(2, 2, figsize=(21 * self.cm, 10 * self.cm),
                                 gridspec_kw={"width_ratios": [3, 2]})

        # Top-left: primitive scores
        ids = self.dataframe["ID"]
        prim_idx = ~np.isnan(ids)
        ids = ids[prim_idx]
        scores = self.dataframe["score"][prim_idx]
        difficulties = self.dataframe["difficulty"][prim_idx]
        axes[0, 0].bar(ids, difficulties, color="lightblue", width=0.6, label="max. score")
        axes[0, 0].bar(ids, scores, color=self.category_colors[-1], width=0.6, label="achieved score")
        axes[0, 0].set_xticks(np.arange(min(ids), max(ids) + 1)[::2])
        axes[0, 0].set_xticklabels(ids.to_numpy(dtype=int)[::2])
        axes[0, 0].set_xlabel("Primitive $i$")
        axes[0, 0].set_ylabel("Score $S_i$")
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Top-right: category scores
        category_scores = [
            self.dataframe.loc[self.dataframe["category"] == c, "score"].sum()
            for c in self.categories
        ]
        category_difficulties = [
            self.dataframe.loc[self.dataframe["category"] == c, "difficulty"].sum()
            for c in self.categories
        ]
        x_pos = np.arange(len(self.categories))
        axes[0, 1].bar(x_pos, category_difficulties, width=0.6, color="lightblue", edgecolor="black")
        axes[0, 1].bar(x_pos, category_scores, width=0.6, color=self.category_colors[-1], edgecolor="black")
        axes[0, 1].set_xlabel("Category")
        axes[0, 1].set_ylabel("Score $S_c$")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(self.categories, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Bottom-left: subtask scores
        subtask_names = list(self.subtask_scores.keys())
        st_scores = [self.subtask_scores[s] for s in subtask_names]
        st_max_scores = [self.subtask_max_scores[s] for s in subtask_names]
        x_pos = np.arange(len(subtask_names))
        axes[1, 0].bar(x_pos, st_max_scores, color="lightblue", width=0.6)
        axes[1, 0].bar(x_pos, st_scores, color=self.category_colors[-1], width=0.6)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(subtask_names, rotation=45, ha='right')
        axes[1, 0].set_xlabel("Subtask")
        axes[1, 0].set_ylabel("Score $S_{st}$")
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Bottom-right: subassembly scores
        subassembly_indices = self.dataframe[self.dataframe["primitive"].str.contains("subassembly")].index
        subassembly_scores = [
            self.dataframe.loc[:sub_idx - 1, "score"].sum() if i == 0
            else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "score"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        subassembly_difficulties = [
            self.dataframe.loc[:sub_idx - 1, "difficulty"].sum() if i == 0
            else self.dataframe.loc[subassembly_indices[i - 1] + 1:sub_idx - 1, "difficulty"].sum()
            for i, sub_idx in enumerate(subassembly_indices)
        ]
        x_pos = np.arange(len(subassembly_indices))
        axes[1, 1].bar(x_pos, subassembly_difficulties, width=0.6, color="lightblue", edgecolor="black")
        axes[1, 1].bar(x_pos, subassembly_scores, width=0.6, color=self.category_colors[-1], edgecolor="black")
        axes[1, 1].set_xlabel("Subassembly")
        axes[1, 1].set_ylabel("Score $S_{sa}$")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f"SA{i+1}" for i in range(len(subassembly_indices))])
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.legend(["max. score", "achieved score"], loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        plt.savefig("07_combined_scores.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_combined_reliability(self):
        """2×2 grid: primitive reliability (top-left), category reliability (top-right),
        subtask reliability (bottom-left), subassembly reliability (bottom-right)."""
        fig, axes = plt.subplots(2, 2, figsize=(21 * self.cm, 10 * self.cm),
                                 gridspec_kw={"width_ratios": [3, 2]})

        # Top-left: primitive reliability
        ids = self.dataframe["ID"]
        prim_idx = ~np.isnan(ids)
        ids = ids[prim_idx]
        reliability = self.dataframe["reliability"][prim_idx] * 100
        axes[0, 0].bar(ids, reliability, color=self.category_colors[-1], width=0.6)
        axes[0, 0].set_xticks(np.arange(min(ids), max(ids) + 1)[::2])
        axes[0, 0].set_xticklabels(ids.to_numpy(dtype=int)[::2])
        axes[0, 0].set_xlabel("Primitive $i$")
        axes[0, 0].set_ylabel("Reliability $R_i$ [%]")
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 110])

        # Top-right: category reliability (boxplot)
        category_reliability = [
            self.dataframe.loc[self.dataframe["category"] == c, "reliability"] * 100
            for c in self.categories
        ]
        bplot = axes[0, 1].boxplot(category_reliability, patch_artist=True)
        for patch, median in zip(bplot['boxes'], bplot['medians']):
            patch.set_facecolor(self.category_colors[-1])
            median.set_color("black")
        axes[0, 1].set_xlabel("Category")
        axes[0, 1].set_xticks(range(1, len(self.categories) + 1))
        axes[0, 1].set_xticklabels(self.categories, rotation=45, ha='right')
        axes[0, 1].set_ylabel("Reliability $R_c$ [%]")
        axes[0, 1].set_ylim([0, 110])

        # Bottom-left: subtask reliability
        subtask_names = list(self.subtask_reliability.keys())
        st_reliabilities = [self.subtask_reliability[s] * 100 for s in subtask_names]
        x_pos = np.arange(len(subtask_names))
        axes[1, 0].bar(x_pos, st_reliabilities, color=self.category_colors[-1], width=0.6)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(subtask_names, rotation=45, ha='right')
        axes[1, 0].set_xlabel("Subtask")
        axes[1, 0].set_ylabel("Reliability $R_{st}$ [%]")
        axes[1, 0].set_ylim([0, 110])
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Bottom-right: subassembly reliability
        subassembly_indices = self.dataframe[self.dataframe["primitive"].str.contains("subassembly")].index
        subassembly_reliability = [
            self.dataframe.loc[subassembly_indices[i], "reliability"] * 100
            for i in range(len(subassembly_indices))
        ]
        axes[1, 1].bar([f"SA{i+1}" for i in range(len(subassembly_indices))],
                       subassembly_reliability, color=self.category_colors[-1], edgecolor="black", width=0.6)
        axes[1, 1].set_xlabel("Subassembly")
        axes[1, 1].set_ylabel("Reliability $R_{sa}$ [%]")
        axes[1, 1].set_ylim([0, 110])

        plt.tight_layout()
        plt.savefig("08_combined_reliability.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def visualize_results(self):
        self.generate_evaluation_protocol()
        self._plot_combined_scores()
        self._plot_combined_reliability()

if __name__ == "__main__":
    asm = AssemblyBenchmark(filepath=r"evaluation\benchmark_protocol_sheet_neu.xlsx")
    asm.evaluate()
    asm.visualize_results()
