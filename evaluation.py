"""
Contains code for the generation of all evaluation figures and tables.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from components.data import DummyData
from testing.testing_pipeline import get_all_training_configs


def main():
    config_parts = get_config_parts()
    task_types = [
        "binary_classification",
        "multiclass_classification",
        "regression",
    ]
    type_results = {}
    for task_type in task_types:
        results = {}
        for file in os.listdir(os.path.join("results", task_type)):
            results[file.replace(".csv", "")] = eval_single_task(
                os.path.join("results", task_type, file), task_type, config_parts
            )
        type_results[task_type] = results
        format_to_latex(create_tabular_aggregations(results, task_type, config_parts))
        format_to_latex(create_all_aggregations(results, task_type, config_parts))
        format_to_latex(create_boosting_aggregations(results, task_type, config_parts))
        format_to_latex(create_standard_deviation(results, task_type, config_parts))
        create_specialist_scatter(
            results, task_type, config_parts, get_metrics(task_type)[-2:]
        )
        stats = pd.read_csv(task_type.split("_")[0] + "_stats.csv", index_col=0)
        master_df = create_master_df(results, stats, config_parts)
        create_heatmap(
           master_df, get_task_stats(task_type), get_metrics(task_type)[-2:]
        )
    format_to_latex(create_summary_aggregations(type_results, task_types, config_parts))


def create_heatmap(master_df: pd.DataFrame, stats: list[str], metrics: list) -> None:
    _, axes = plt.subplots(
        1, len(metrics), figsize=(7 * len(metrics) * 0.7, 3 * 0.7), squeeze=False
    )
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        results = []
        for m_type, group in master_df.groupby("config_type"):
            if m_type == "Tabular":
                continue
            if "Boosted" in str(m_type):
                b_corrs = group[stats + ["diff_" + metric]].corr(method="spearman")[
                    "diff_" + metric
                ]
                b_corrs = b_corrs.drop("diff_" + metric)
                b_corrs.name = m_type + " G"
                results.append(b_corrs)
        for m_type, group in master_df.groupby("config_type"):
            if m_type == "Tabular":
                continue
            corrs = group[stats + [metric]].corr(method="spearman")[metric]
            corrs = corrs.drop(metric)
            corrs.name = m_type + " Z"
            results.append(corrs)
        heatmap_df = pd.concat(results, axis=1).T
        heatmap_df.rename(get_stat_rename_dict(), inplace=True, axis=1)
        yticklabels = i == 0
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            yticklabels=yticklabels,
            vmin=-0.5,
            vmax=0.5,
        )
        ax.set_title(improve_metrics(metric), fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def create_master_df(results: dict, stats: pd.DataFrame, config_parts) -> pd.DataFrame:
    stats["task_name"] = stats["dataset"] + "-" + stats["task"]
    task_dfs = []
    for task, df in results.items():
        task_df = df.copy(deep=True)
        task_df["task_name"] = [task] * len(task_df.index)
        task_dfs.append(task_df)
    results_df = pd.concat(task_dfs, ignore_index=True)
    master_df = pd.merge(results_df, stats, on="task_name")
    master_df["config_type"] = master_df["name"].apply(
        get_config_category, (config_parts,)
    )
    return master_df


def get_config_category(config: str, config_parts: dict) -> str:
    if "Tabular" in config:
        return "Tabular"
    if len(config_parts[config]) == 1:
        return "Single"
        if "shallow" in config:
            return "Single-shallow"
        if "medium" in config:
            return "Single-medium"
        if "deep" in config:
            return "Single-deep"
    else:
        return "Boosted"
        if "+" in config:
            return "Dual-boost"
        elif "All" in config:
            return "All-boost"
        elif "rev" in config:
            return "Model-rev-boost"
        else:
            return "Model-boost"


def create_specialist_scatter(
    data: dict, task_type: str, config_parts: dict, metrics: list
) -> None:
    aggr = create_all_aggregations(data, task_type, config_parts)
    std = create_standard_deviation(data, task_type, config_parts)
    types = [
        "single" if len(config_parts[config]) == 1 else "boosted"
        for config in list(aggr.index)
    ]
    sns.set_theme(style="whitegrid")
    _, axes = plt.subplots(
        1, len(metrics), figsize=(6.5 * len(metrics) * 0.75, 7 * 0.75), squeeze=False
    )
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        df = pd.DataFrame(
            {
                "Config": aggr.index,
                "Mean Z-score": aggr[metric].values,
                "Z-score Std.Dev.": std[metric].values,
                "Type": types,
            }
        )
        points = sns.scatterplot(
            data=df,
            x="Mean Z-score",
            y="Z-score Std.Dev.",
            hue="Type",
            style="Type",
            s=180,
            edgecolor="black",
            ax=ax,
        )
        ax.set_title(improve_metrics(metric), fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        if metric == "f1":
            texts = []
            for i in get_scatter_pareto(df):
                texts.append(
                    ax.text(
                        df.loc[i, "Mean Z-score"],
                        df.loc[i, "Z-score Std.Dev."],
                        shrink_config_names(df.loc[i, "Config"]),
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="white", alpha=0.5, pad=0.0)
                    )
                )
            adjust_text(
                texts,
                ax=ax,
                x=df["Mean Z-score"],
                y=df["Z-score Std.Dev."],
                #force_text=(2.0, 1.0),
                ensure_inside_axes=False,
                expand=(2.0, 1.2),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            )
        else:
            for i in get_scatter_pareto(df):
                ax.annotate(
                    shrink_config_names(df.loc[i, "Config"]),
                    xy=(df.loc[i, "Mean Z-score"], df.loc[i, "Z-score Std.Dev."]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    fontsize=10,
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.5, pad=0.0),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
                )
    plt.tight_layout()
    plt.show()


def get_scatter_pareto(df: pd.DataFrame) -> list:
    result = []
    x = "Mean Z-score"
    y = "Z-score Std.Dev."

    for i, config in df.iterrows():
        dominated = False
        for j, comp in df.iterrows():
            if i == j:
                continue
            if comp[x] >= config[x] and comp[y] <= config[y]:
                dominated = True
                break
        if not dominated:
            result.append(i)
    return result


def shrink_config_names(config: str) -> str:
    return (
        config.replace("GraphSAGE", "GS")
        .replace("RelGNN", "RG")
        .replace("DBFormer", "DB")
        .replace("RelGT", "GT")
        .replace("rev_boost", "rb")
        .replace("boost", "b")
        .replace("shallow", "s")
        .replace("medium", "m")
        .replace("deep", "d")
    )


def format_to_latex(df: pd.DataFrame) -> str:
    result = "\\begin{tabular}{" + " ".join(["l"] * (len(df.columns) + 1)) + "}\n"
    result += "\t\\uzlhline"
    for col in list(df.columns.values):
        result += "& \\uzlemph{" + improve_metrics(col) + "} "
    result += "\\\\\\uzlhline\n"
    for config in df.index:
        result += "\t" + config + " "
        for column in df:
            val = df.loc[config, column]
            if type(val) == int:
                result += f"& {val!s} "
            elif type(val) == np.float64:
                result += f"& {round(val, 2)!s} "
            else:
                result += f"& {val} "
        result += "\\\\\n"
    result += "\t\\uzlhline\n"
    result += "\\end{tabular}"
    print(result.replace("_", "\\_"))


def create_summary_aggregations(
    type_data: dict, task_types: list, config_parts: dict
) -> pd.DataFrame:
    results = pd.DataFrame(columns=["config", "boosted", "performance", "instability"])
    configs = []
    for config in config_parts:
        if "Tabular" not in config:
            configs.append(config)
    results["config"] = configs
    results = results.set_index("config")
    type_means = {}
    type_deviations = {}
    for task_type in task_types:
        metrics = get_metrics(task_type)
        for config in configs:
            means = []
            deviations = []
            for metric in metrics:
                values = []
                for df in type_data[task_type].values():
                    if config not in df["name"].array:
                        continue
                    values.append(df.loc[df["name"] == config, metric].item())
                means.append(np.mean(values))
                deviations.append(np.std(values))
            if config not in type_means:
                type_means[config] = []
                type_deviations[config] = []
            type_means[config].append(np.mean(means))
            type_deviations[config].append(np.mean(deviations))
    for config in configs:
        results.loc[config, "boosted"] = (
            "\\checkmark" if len(config_parts[config]) > 1 else ""
        )
        results.loc[config, "performance"] = np.mean(type_means[config])
        results.loc[config, "instability"] = np.mean(type_deviations[config])
    return results.sort_values(by="performance", ascending=False)


def create_single_aggregations(
    data: dict, task_type: str, config_parts: dict
) -> pd.DataFrame:
    metrics = get_metrics(task_type)
    results = pd.DataFrame(columns=["config", "count"] + metrics)
    configs = []
    for config, parts in config_parts.items():
        if len(parts) == 1:
            configs.append(config)
    results["config"] = configs
    results = results.set_index("config")
    for config in configs:
        for metric in metrics:
            values = []
            for df in data.values():
                if config not in df["name"].array:
                    continue
                values.append(df.loc[df["name"] == config, metric].item())
            results.loc[config, metric] = np.mean(values)
        results.loc[config, "count"] = len(values)
    return results


def create_all_aggregations(
    data: dict, task_type: str, config_parts: dict
) -> pd.DataFrame:
    metrics = get_metrics(task_type)
    results = pd.DataFrame(columns=["config", "count"] + metrics)
    configs = []
    for config in config_parts:
        if "Tabular" not in config:
            configs.append(config)
    results["config"] = configs
    results = results.set_index("config")
    for config in configs:
        for metric in metrics:
            values = []
            for df in data.values():
                if config not in df["name"].array:
                    continue
                values.append(df.loc[df["name"] == config, metric].item())
            results.loc[config, metric] = np.mean(values)
        results.loc[config, "count"] = len(values)
    return results


def create_boosting_aggregations(
    data: dict, task_type: str, config_parts: dict
) -> pd.DataFrame:
    metrics = get_metrics(task_type)
    results = pd.DataFrame(columns=["config", "count"] + metrics)
    configs = []
    for config, parts in config_parts.items():
        if len(parts) > 1 and "Tabular" not in config:
            configs.append(config)
    results["config"] = configs
    results = results.set_index("config")
    for config in configs:
        for metric in metrics:
            values = []
            for df in data.values():
                if config not in df["name"].array:
                    continue
                values.append(df.loc[df["name"] == config, "diff_" + metric].item())
            results.loc[config, metric] = np.mean(values)
        results.loc[config, "count"] = len(values)
    return results


def create_tabular_aggregations(
    data: dict, task_type: str, config_parts: dict
) -> pd.DataFrame:
    metrics = get_metrics(task_type)[-2:]
    columns = ["config", "count"]
    for metric in metrics:
        columns.append("tab_" + metric)
        columns.append("boost_" + metric)
    results = pd.DataFrame(columns=columns)
    configs = []
    for config, parts in config_parts.items():
        if len(parts) > 1 and "Tabular" in config:
            configs.append(config)
    results["config"] = configs
    results = results.set_index("config")
    for config in configs:
        for metric in metrics:
            tab_values = []
            boost_values = []
            for df in data.values():
                if config not in df["name"].array:
                    continue
                tab_values.append(
                    df.loc[df["name"] == config, "tab_tab_" + metric].item()
                )
                boost_values.append(
                    df.loc[df["name"] == config, "tab_boost_" + metric].item()
                )
            results.loc[config, "tab_" + metric] = np.mean(tab_values)
            results.loc[config, "boost_" + metric] = np.mean(boost_values)
        results.loc[config, "count"] = len(tab_values)
    return results


def create_standard_deviation(
    data: dict, task_type: str, config_parts: dict
) -> pd.DataFrame:
    metrics = get_metrics(task_type)
    results = pd.DataFrame(columns=["config", "count"] + metrics)
    configs = []
    for config in config_parts:
        if "Tabular" not in config:
            configs.append(config)
    results["config"] = configs
    results = results.set_index("config")
    for config in configs:
        for metric in metrics:
            values = []
            for df in data.values():
                if config not in df["name"].array:
                    continue
                values.append(df.loc[df["name"] == config, metric].item())
            results.loc[config, metric] = np.std(values)
        results.loc[config, "count"] = len(values)
    return results


def eval_single_task(path: str, task_type: str, config_parts: dict) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    cols_to_transform = get_metrics(task_type)
    for col in cols_to_transform:
        lower_is_better = col in ["mae", "rmse"]
        df[col] = z_score(df[col], lower_is_better=lower_is_better)

    for metric in cols_to_transform:
        values = []
        for row in df["name"]:
            parts = config_parts[row]
            max_value = df.loc[df["name"] == parts[0], metric].item()
            for part in parts:
                max_value = max(max_value, df.loc[df["name"] == part, metric].item())
            values.append(df.loc[df["name"] == row, metric].item() - max_value)
        df["diff_" + metric] = values

        tab_tab = []
        tab_boost = []
        for row in df["name"]:
            if "Tabular+" not in row:
                tab_tab.append(0.0)
                tab_boost.append(0.0)
                continue
            parts = config_parts[row]
            tab_tab.append(
                df.loc[df["name"] == row, metric].item()
                - df.loc[df["name"] == parts[0], metric].item()
            )
            tab_boost.append(
                df.loc[df["name"] == row, metric].item()
                - df.loc[df["name"] == parts[1], metric].item()
            )
        df["tab_tab_" + metric] = tab_tab
        df["tab_boost_" + metric] = tab_boost

    return df


def get_metrics(task_type: str) -> str:
    if task_type == "binary_classification":
        return ["accuracy", "average_precision", "f1", "roc_auc"]
    if task_type == "multiclass_classification":
        return ["mrr", "accuracy", "macro_f1"]
    if task_type == "regression":
        return ["r2", "rmse", "mae"]


def get_task_stats(task_type: str) -> str:
    base = ["n_train", "unique", "eccentricity", "avg_shortest_path"]
    if task_type == "binary_classification":
        return base + ["balance"]
    if task_type == "multiclass_classification":
        return ["n_train", "n_classes", "freq_ratio", "hhi"]
    if task_type == "regression":
        return base + ["skewness", "kurtosis"]


def improve_metrics(metric: str) -> str:
    improvements = {
        "average_precision": "Avg.Pr.",
        "accuracy": "Acc.",
        "f1": "F1",
        "roc_auc": "AUROC",
        "macro_f1": "Mac.F1",
        "mrr": "MRR",
        "r2": "R2",
        "rmse": "RMSE",
        "mae": "MAE",
        "count": "n",
    }
    if metric not in improvements:
        if "tab_" in metric:
            return improve_metrics(metric.replace("tab_", "")) + "(Tab)"
        if "boost_" in metric:
            return improve_metrics(metric.replace("boost_", "")) + "(Boost)"
        return metric
    return improvements[metric]


def get_stat_rename_dict() -> dict:
    return {
        "n_train": "N",
        "unique": "Unique",
        "eccentricity": "Ecc.",
        "avg_shortest_path": "ASP",
        "balance": "Balance",
        "freq_ratio": "Ratio",
        "hhi": "HHI",
        "skewness": "Skew.",
        "kurtosis": "Kurt.",
        "n_classes": "C",
    }


def z_score(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    test = series.to_numpy()
    if (test[0] == test).all():
        return pd.Series([0.0 for _ in range(len(series))])

    mean = series.mean()
    std = series.std(ddof=0)
    if not lower_is_better:
        return (series - mean) / std
    else:
        return (mean - series) / std


def get_config_parts() -> dict:
    configs = get_all_training_configs(DummyData())
    parts = {}
    for name in configs:
        if "Tabular+" in name:
            parts[name] = ["Tabular", name.replace("Tabular+", "")]
            continue
        parts[name] = []
        for config in configs[name]:
            if config["name"] not in parts[name]:
                parts[name].append(config["name"])
    return parts


if __name__ == "__main__":
    main()

    # Planned Metrics:
    # Aggregated z-scores per config?
    # For boosting: improvement over best of base
