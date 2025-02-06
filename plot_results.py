import csv
from textwrap import wrap

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# COLOR & STYLE UTILITIES
###############################################################################

# 8 color set (ColorBrewer Set1-like):
subtask_color_map = {
    "AST": "#E41A1C",
    "FBT": "#377EB8",
    "HTT": "#4DAF4A",
    "FPR": "#984EA3",
    "UOT": "#FF7F00",
    "PST": "#FFFF33",
    "SST": "#A65628",
    "SIT": "#F781BF",
}

# Special color for raw model
raw_color = "#333333"


def adjust_color(color_hex, factor=0.7):
    """
    Lighten or darken a color by a given factor.
      factor < 1 => darker
      factor > 1 => lighter
    """
    rgb = mcolors.to_rgb(color_hex)
    new_rgb = tuple(min(max(c * factor, 0.0), 1.0) for c in rgb)
    return mcolors.to_hex(new_rgb)


def get_bar_edge_style(model_name):
    """
    Return (edgecolor, hatch_line) to differentiate e.g. 3.1 vs 3.2
    You can customize as you see fit.
    """
    if "3.1-8b" in model_name:
        # e.g. dashed edge
        return "dashed"
    elif "3.2-3b" in model_name:
        # e.g. dotted edge
        return "dotted"
    else:
        # fallback
        return "solid"


# Added utility functions to resolve the missing definitions:
def final_sort_key(d):
    return (
        d["type"] != "raw",
        d["pruned_on"] if d["pruned_on"] is not None else "",
        float(d["sparsity"].replace("%", "")) if "%" in d["sparsity"] else 0,
        1 if "3.2" in d["model_name"] else 0,
    )


def get_bar_color(d):
    if d["type"] == "raw":
        return raw_color
    subtask = d["pruned_on"]
    base_color = subtask_color_map.get(subtask, "#666666")
    sp_float = float(d["sparsity"].replace("%", "")) if "%" in d["sparsity"] else 0
    if sp_float == 25.0:
        return adjust_color(base_color, factor=1.2)
    elif sp_float == 50.0:
        return adjust_color(base_color, factor=0.8)
    else:
        return base_color


###############################################################################
# PLOTTING: TOM EVAL
###############################################################################


def get_model_label_color(model):
    if model["type"] == "raw":
        return raw_color
    base = subtask_color_map.get(model["pruned_on"], "#666666")
    try:
        sp = float(model["sparsity"].replace("%", ""))
    except:
        sp = 0
    if sp == 25.0:
        return adjust_color(base, 1.2)
    elif sp == 50.0:
        return adjust_color(base, 0.8)
    else:
        return base


def plot_tom_eval(
    csv_file="results/tom_eval.csv", output_png="plots/tom_eval_heatmaps.png"
):
    """
    Creates two line plots from tom_eval.csv:
      - Left: Llama-3.1-8b models.
      - Right: Llama-3.2-3b models.
    For each plot, x-axis represents ordered models and each line corresponds to a subtask.
    Lines use subtask colors defined in subtask_color_map and a horizontal legend is placed on top.
    """
    subtasks = ["AST", "FBT", "HTT", "FPR", "UOT", "PST", "SST", "SIT"]

    # Read CSV and build data per model
    models_31, models_32 = [], []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = {
                "model_name": r["name"],
                "type": r["type"],
                "pruned_on": (
                    r["pruned_on"] if r["pruned_on"] not in ("None", "") else None
                ),
                "sparsity": r["sparsity"],
                "scores": {},
            }
            for s in subtasks:
                val = r[s].strip()
                if val == "":
                    d["scores"][s] = np.nan
                else:
                    try:
                        d["scores"][s] = float(val)
                    except ValueError:
                        d["scores"][s] = np.nan
            if "3.1-8b" in d["model_name"]:
                models_31.append(d)
            elif "3.2-3b" in d["model_name"]:
                models_32.append(d)

    def parse_sparsity(sp):
        try:
            return float(sp.replace("%", ""))
        except:
            return 999

    def sort_key(d):
        return (d["type"] != "raw", d["pruned_on"] or "", parse_sparsity(d["sparsity"]))

    models_31.sort(key=sort_key)
    models_32.sort(key=sort_key)

    # Create figure with 2 subplots for line plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, models, title in [
        (ax1, models_31, "Cross-Subtask Pruning Accuracy - Llama-3.1-8b-Instruct"),
        (ax2, models_32, "Cross-Subtask Pruning Accuracy - Llama-3.2-3b-Instruct"),
    ]:
        n_models = len(models)
        x_indices = np.arange(n_models)
        # Plot one line per subtask using its tom subtask color.
        for s in subtasks:
            y_values = [model["scores"].get(s, np.nan) for model in models]
            ax.plot(
                x_indices,
                y_values,
                marker="o",
                label=s,
                color=subtask_color_map.get(s, "#666666"),
            )
        # Set x-axis labels to display model-specific label.
        xticklabels = [
            model.get(
                "label",
                (
                    "Raw"
                    if model["type"] == "raw"
                    else f"{model['pruned_on']} ({model['sparsity']})"
                ),
            )
            for model in models
        ]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
        # Set x-axis label colors based on model properties
        for tick, model in zip(ax.get_xticklabels(), models):
            tick.set_color(get_model_label_color(model))
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        # Removed per-plot legend
    # Add a single, global legend at the top
    leg_handles = [
        mpatches.Patch(color=subtask_color_map.get(s, "#666666"), label=s)
        for s in subtasks
    ]
    fig.legend(
        handles=leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(subtasks),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved line plots in {output_png}")


###############################################################################
# PLOTTING: EVAL ZERO-SHOT SUMMARY
###############################################################################


def plot_eval_zero_shot_summary(
    csv_file="results/eval_zero_shot_summary.csv", output_png="plots/eval_zero_shot.png"
):
    """
    Creates a figure with 2 subplots side-by-side:
    - Left (3/4 width): grouped bar chart for tasks that are accuracy-based
      (hellaswag, winogrande, arc_easy, boolq, arc_challenge).
    - Right (1/4 width): table of perplexities for wikitext.

    Only uses 3b models (since user said we only have them for zero_shot).
    The CSV columns are: model_type,model_name,pruned_on,sparsity,hellaswag,winogrande,arc_easy,boolq,arc_challenge,wikitext
    We'll parse the accuracy tasks as float. The wikitext will be perplexity.
    """
    # read CSV
    tasks_acc = ["hellaswag", "winogrande", "arc_easy", "boolq", "arc_challenge"]
    perplexity_task = "wikitext"

    data_rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            model_name = r["model_name"]
            # skip if not "3b"
            if "3b" not in model_name:
                continue
            data_rows.append(r)

    # We'll have a set of bars for each row, one bar per accuracy task.
    # The color depends on pruned_on and sparsity. If raw => raw color.
    # The edge style depends on model_name (though presumably they're all "3.2-3b"?).
    # We'll store a structure:
    # "label" for the row, a short label, e.g. "raw", "AST(25)", etc.
    # plus a  list of accuracies for tasks_acc
    # plus perplexity for wikitext
    plot_data = []

    def parse_float(s):
        try:
            return float(s)
        except:
            return None

    for row in data_rows:
        d = {}
        d["type"] = row["model_type"]
        d["model_name"] = row["model_name"]
        pruned_on = row["pruned_on"] if row["pruned_on"] not in ("None", "") else None
        d["pruned_on"] = pruned_on
        sp = row["sparsity"]
        d["sparsity"] = sp.replace("pct", "%") if "pct" in sp else sp
        # build a label, e.g. "AST (25%)", or "Raw"
        if row["model_type"] == "raw":
            d["label"] = "Raw"
        else:
            # e.g. AST (25%)
            d["label"] = f"{pruned_on} ({sp})"
        # parse tasks
        d["accs"] = []
        for t in tasks_acc:
            val = parse_float(row[t])
            d["accs"].append(val if val is not None else 0.0)
        # parse perplex
        pval = parse_float(row[perplexity_task])
        d["perplexity"] = pval if pval is not None else None

        plot_data.append(d)

    # We define a stable sort: raw first, then alphabetical pruned_on, then 25% < 50%.
    def parse_sparsity(sp):
        if sp in ("0%", "0.0%"):
            return 0
        try:
            return float(sp.replace("%", ""))
        except:
            return 999

    def sort_key(d):
        return (
            d["type"] != "raw",
            d["pruned_on"] if d["pruned_on"] else "",
            parse_sparsity(d["sparsity"]),
        )

    plot_data_sorted = sorted(plot_data, key=sort_key)

    # Now we create the figure with 2 subplots, using gridspec:
    fig = plt.figure(figsize=(14, 6))  # Slightly wider figure
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[3, 1],
        figure=fig,
        wspace=0.5,  # Reduced spacing between plots
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # Left: grouped bar chart for tasks_acc
    # We'll have len(tasks_acc) groups on the x-axis. We have len(plot_data_sorted) bars for each group
    x_indices = np.arange(len(tasks_acc))
    bar_width = 0.8 / max(len(plot_data_sorted), 1)

    # We'll place each row's bars next to each other.
    for i, d in enumerate(plot_data_sorted):
        offset = (i - (len(plot_data_sorted) - 1) / 2) * bar_width
        # color
        if d["type"] == "raw":
            bar_color = raw_color
        else:
            # Use base pruned color without adjustment
            bar_color = get_bar_color(d)
        # Use a fixed solid edge style for all (only 3b models)
        ls = "solid"
        yvals = d["accs"]  # 5 values
        ax_left.bar(
            x_indices + offset,
            yvals,
            bar_width,
            color=bar_color,
            edgecolor="black",
            linestyle=ls,
            alpha=0.9,
            label=d["label"] if i == 0 else None,
        )

    # We'll do a custom legend at the bottom or so
    # Label the x-axis with tasks:
    ax_left.set_xticks(x_indices)
    # maybe wrap them
    new_labels = ["\n".join(wrap(t, 13)) for t in tasks_acc]
    ax_left.set_xticklabels(new_labels)
    ax_left.set_ylim([0, 1.0])  # assume accuracy between 0 and 1
    ax_left.set_title("Zero-Shot Accuracy on General Benchmarks")

    # Build legend elements (legend_elems) as before:
    from matplotlib.lines import Line2D

    legend_elems = []
    legend_elems.append(Line2D([0], [0], color=raw_color, lw=10, label="Raw"))
    pruned_on_unique = sorted(set(d["pruned_on"] for d in plot_data if d["pruned_on"]))
    for st in pruned_on_unique:
        base_c = subtask_color_map.get(st, "#666666")
        c25 = adjust_color(base_c, 1.2)
        c50 = adjust_color(base_c, 0.8)
        legend_elems.append(Line2D([0], [0], color=c25, lw=10, label=f"{st} (25%)"))
        legend_elems.append(Line2D([0], [0], color=c50, lw=10, label=f"{st} (50%)"))

    # Remove the previous ax_left.legend call and add a global legend placed vertically between subplots.
    fig.legend(
        handles=legend_elems,
        loc="center",
        bbox_to_anchor=(0.68, 0.5),  # Adjusted for new spacing
        ncol=1,
        borderaxespad=0,
    )

    # Right: horizontal bar plot for wikitext perplexity
    ax_right.set_axis_on()
    n_models = len(plot_data_sorted)
    y_positions = np.arange(n_models)
    ppl_values = []
    colors = []
    labels = []
    for d in plot_data_sorted:
        ppl = d["perplexity"]
        ppl_values.append(ppl if ppl is not None else 0)
        labels.append(d["label"])
        if d["type"] == "raw":
            colors.append(raw_color)
        else:
            base_color = subtask_color_map.get(d["pruned_on"], "#666666")
            sp = float(d["sparsity"].replace("%", "")) if "%" in d["sparsity"] else 0
            if sp == 25.0:
                colors.append(adjust_color(base_color, 1.2))
            elif sp == 50.0:
                colors.append(adjust_color(base_color, 0.8))
            else:
                colors.append(base_color)

    ax_right.barh(y_positions, ppl_values, color=colors, edgecolor="black")
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels([])  # removed labels; legend already shows model info
    ax_right.invert_yaxis()  # optional: first entry on top
    ax_right.set_xlabel("Perplexity")
    ax_right.set_title("Wikitext Perplexity")

    plt.tight_layout()
    # Adjust subplot parameters to better fill the space
    plt.subplots_adjust(
        left=0.05,  # Reduced left margin
        right=0.95,  # Increased right margin
        bottom=0.15,  # Space for x-labels
        top=0.9,  # Space for titles
        wspace=0.5,  # Match GridSpec wspace
    )
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved figure {output_png}")


def plot_layerwise_weight_distribution_1(
    csv_file="results/layerwise_weight_distribution.csv",
    output_png="plots/layerwise_dist_summary.png",
):
    """
    This function computes the *average fraction of nonzero parameters* across
    different "component types" (embed, attn, mlp, norm) for each pruned model,
    and displays them as a grouped bar chart.
    It's a compact way to see, for each model, how many parameters remain in
    embeddings vs. self-attention vs. MLP vs. layernorm, on average.
    """
    import csv
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # We’ll categorize each 'layer' name into a broad component:
    def categorize_layer(layer_name):
        if "embed_tokens" in layer_name:
            return "Embedding"
        elif ".self_attn." in layer_name:
            return "Attention"
        elif ".mlp." in layer_name:
            return "MLP"
        elif "norm" in layer_name:
            return "LayerNorm"
        else:
            return "Other"

    # Read the CSV into a list of dicts
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields
            r["total_params"] = float(r["total_params"])
            r["nonzero_params"] = float(r["nonzero_params"])
            rows.append(r)

    # Group by (type, name, pruned_on, sparsity, component)
    # Then we’ll average fraction_remaining = nonzero/total across layers in that group
    from collections import defaultdict

    group_data = defaultdict(lambda: {"sum_fraction": 0.0, "count": 0})

    def group_key(r):
        ctype = categorize_layer(r["layer"])
        return (r["type"], r["name"], r["pruned_on"], r["sparsity"], ctype)

    for r in rows:
        key = group_key(r)
        fraction = (
            (r["nonzero_params"] / r["total_params"]) if r["total_params"] > 0 else 0
        )
        group_data[key]["sum_fraction"] += fraction
        group_data[key]["count"] += 1

    # Now we build a list of “models” => (type,name,pruned_on,sparsity) in stable order
    # Then for each, we have 4-5 component categories (Embedding, Attention, MLP, LayerNorm, Other).
    from collections import OrderedDict

    models_info = (
        OrderedDict()
    )  # (type,name,pruned_on,sparsity) -> dict of component -> fraction

    def parse_sparsity(sp):
        try:
            return float(sp.replace("%", ""))
        except:
            return 999

    def model_sort_key(t, n, p, sp):
        return (
            t != "raw",  # raw first
            p if p else "",
            parse_sparsity(sp),
            1 if "3.2" in n else 0,
        )

    # Accumulate a set of all component categories:
    all_components = set()
    for (t, n, p, sp, comp), valdic in group_data.items():
        all_components.add(comp)

    # Then gather each model & fill the data:
    # model_key -> {comp: fraction}
    for (t, n, p, sp, comp), valdic in group_data.items():
        mk = (t, n, p, sp)
        if mk not in models_info:
            models_info[mk] = {}
        avg_frac = valdic["sum_fraction"] / valdic["count"]
        models_info[mk][comp] = avg_frac

    # Sort the model_keys:
    sorted_model_keys = sorted(models_info.keys(), key=lambda mk: model_sort_key(*mk))

    # We'll define an explicit list for the components, in a nice order
    component_order = ["Embedding", "Attention", "MLP", "LayerNorm", "Other"]
    # Remove if not present in all_components:
    component_order = [c for c in component_order if c in all_components]

    # We'll do a grouped bar chart: x-axis => each model, groups => each model
    # inside each group => a bar per component.
    x_indices = np.arange(len(sorted_model_keys))
    bar_width = 0.8 / len(component_order)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, comp in enumerate(component_order):
        offsets = x_indices + (i - (len(component_order) - 1) / 2) * bar_width
        yvals = []
        for mk in sorted_model_keys:
            fraction_val = models_info[mk].get(comp, 0.0)
            yvals.append(fraction_val)
        ax.bar(offsets, yvals, width=bar_width, label=comp, alpha=0.8)

    # Build x tick labels from the model info
    xticks_labels = []
    for mk in sorted_model_keys:
        t, n, p, sp = mk
        if t == "raw":
            xticks_labels.append("Raw")
        else:
            # e.g. AST(25%)
            label_str = f"{p} ({sp})"
            xticks_labels.append(label_str)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(xticks_labels, rotation=45, ha="right")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of Nonzero Params")
    ax.set_title("Avg Fraction of Nonzero Params by Component")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved plot_layerwise_weight_distribution_1 -> {output_png}")


def plot_layerwise_weight_distribution_2(
    csv_file="results/layerwise_weight_distribution.csv",
    output_png="plots/layerwise_dist_boxplot.png",
):
    """
    This function constructs a boxplot of the 'mean' (average parameter value)
    distribution across *all layers* for each model, allowing us to compare how
    pruning might shift parameter means. We group boxplots by
    (pruned_on, sparsity) plus a single box for raw models.
    """
    import csv
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # Read the CSV
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # parse numeric
            try:
                r["mean"] = float(r["mean"])
            except:
                r["mean"] = 0
            rows.append(r)

    # Collect a dictionary: model_key -> list of param means
    from collections import defaultdict

    model_means = defaultdict(list)

    def parse_sparsity(sp):
        try:
            return float(sp.replace("%", ""))
        except:
            return 999

    for r in rows:
        model_type = r["type"]
        if model_type == "raw":
            mk = ("raw", "raw")
        else:
            pruned_on = r["pruned_on"]
            sp = r["sparsity"]
            mk = (pruned_on, sp)
        model_means[mk].append(r["mean"])

    # We’ll create a stable sorted list of keys:
    # raw first, then alphabetical pruned_on, then ascending sparsity
    def sort_key(mk):
        pruned_on, spt = mk
        if pruned_on == "raw":
            return (False, "", 0)
        else:
            return (True, pruned_on, parse_sparsity(spt))

    sorted_mkeys = sorted(model_means.keys(), key=sort_key)

    # Build boxplot data
    box_data = []
    xtick_labels = []
    for mk in sorted_mkeys:
        means_list = model_means[mk]
        box_data.append(means_list)
        if mk[0] == "raw":
            xtick_labels.append("Raw")
        else:
            xtick_labels.append(f"{mk[0]}({mk[1]})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(box_data, labels=xtick_labels, showfliers=False, vert=True)
    ax.set_title("Distribution of Parameter Means per Model")
    ax.set_ylabel("Parameter 'mean' across layers")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved plot_layerwise_weight_distribution_2 -> {output_png}")


def plot_layerwise_l2norm_ratio(
    csv_file="results/layerwise_weight_distribution.csv",
    output_png="plots/layerwise_l2norm_ratio.png",
):
    """
    Plots how much the L2 norm per layer changes, relative to the raw baseline,
    for each pruned model. For each layer, we compute:
        ratio = (l2_norm of pruned) / (l2_norm of the raw model)
    and then plot ratio vs. layer index as a line plot.

    Even with uniform Wanda pruning, the absolute weight distribution can differ
    from layer to layer, so these ratio curves may still reveal interesting
    subtask- or sparsity-specific differences across the layers.
    """
    import csv
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np

    # We'll store:  model_key = (type, name, pruned_on, sparsity)
    # Then inside:  layer_l2[layer_index or layer_name] = l2_norm
    # We'll need to find the corresponding "raw" model for each pruned model
    # typically type=raw, pruned_on=None, same "name".
    # We'll define layer indexing logic: we parse the "layer" column. If
    # it has "layers.0." => layer_index=0, etc. If there's "embed_tokens" => -1?
    # We'll keep it simple: just store the entire string, but we can't line-plot well
    # unless we parse an integer index. Let's parse "model.layers.X." if it exists.

    def extract_layer_index(layer_str):
        # e.g. "model.layers.5.self_attn.q_proj.weight" => 5
        # if none found, return None
        import re

        match = re.search(r"\.layers\.(\d+)\.", layer_str)
        if match:
            return int(match.group(1))
        else:
            return None

    # Read the CSV
    all_data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse numeric
            row["l2_norm"] = float(row["l2_norm"]) if row["l2_norm"] else 0
            # store layer_index
            row["layer_index"] = extract_layer_index(row["layer"])
            all_data.append(row)

    # Build a dict: model -> { layer_index -> l2_norm }
    # model here is a 4-tuple (type, name, pruned_on, sparsity)
    from collections import defaultdict

    model_to_layer2norm = defaultdict(dict)

    def make_model_key(r):
        return (r["type"], r["name"], r["pruned_on"], r["sparsity"])

    for r in all_data:
        mk = make_model_key(r)
        idx = r["layer_index"]
        if idx is not None:
            model_to_layer2norm[mk][idx] = r["l2_norm"]

    # We also want to find "raw" baselines, i.e. (type='raw', name=?).
    # We'll do: raw_key => model_to_layer2norm, pruned_key => ratio = pruned / raw
    # We'll store a new structure: { pruned_key -> {layer_index: ratio} }
    ratio_data = defaultdict(dict)

    # first gather raw models by name
    raw_models_by_name = {}
    for mk in model_to_layer2norm.keys():
        (mtype, mname, pon, sp) = mk
        if mtype == "raw":
            # treat sp as e.g. '0%' or '' etc
            # store in raw_models_by_name[(mname)] = mk
            raw_models_by_name[mname] = mk

    # Now for each pruned, find ratio
    for mk in model_to_layer2norm.keys():
        (mtype, mname, pon, sp) = mk
        if mtype == "pruned":
            # attempt to find raw
            if mname in raw_models_by_name:
                raw_mk = raw_models_by_name[mname]
                l2map_pruned = model_to_layer2norm[mk]
                l2map_raw = model_to_layer2norm[raw_mk]
                # gather the union of layer indices
                all_idx = sorted(set(l2map_pruned.keys()) & set(l2map_raw.keys()))
                for idx in all_idx:
                    if l2map_raw[idx] != 0:
                        ratio = l2map_pruned[idx] / l2map_raw[idx]
                    else:
                        ratio = 0
                    ratio_data[mk][idx] = ratio
        else:
            # if it's raw, we might skip or store a degenerate ratio=1 for reference
            pass

    # Now we can plot. We'll do a figure with one axis. We'll line-plot each pruned_on
    # We might have many lines (8 subtasks * 2 sparsities). We'll store them in sets.
    import math

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # We want to group the pruned keys by (pruned_on, sparsity)
    # because we might have multiple model names (like 3.1 vs 3.2).
    # but let's keep it simple. We'll just plot them all. If you have duplicates,
    # it might appear multiple times. We can reduce by set((pon, sp)).
    # We'll define a stable sort so "AST(25%)" etc. come in some order.

    def parse_sparsity(sp):
        try:
            return float(sp.replace("%", ""))
        except:
            return 0

    # gather unique subtask-sparsities
    curves = []
    for mk in ratio_data.keys():
        (mtype, mname, pon, sp) = mk
        if pon in (None, "", "None"):
            continue
        # build label
        label_str = f"{pon}({sp}) {mname}"
        # gather layer indexes + ratio
        layer_idx_sorted = sorted(ratio_data[mk].keys())
        yvals = [ratio_data[mk][i] for i in layer_idx_sorted]
        curves.append(
            (pon, parse_sparsity(sp), mname, layer_idx_sorted, yvals, label_str)
        )

    # sort curves
    curves.sort(key=lambda x: (x[0], x[1], x[2]))

    # Now plot each
    for c in curves:
        layer_idx_sorted, yvals, label_str = c[3], c[4], c[5]
        ax.plot(layer_idx_sorted, yvals, marker="o", label=label_str, alpha=0.8)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("L2 Norm (pruned) / L2 Norm (raw)")
    ax.set_title("Layerwise L2 Norm Ratio (pruned ÷ raw)")
    ax.set_ylim(0, None)  # allow >1 if it occurs, though typically <1
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved L2 norm ratio plot to {output_png}")


def plot_layerwise_std_ratio(
    csv_file="results/layerwise_weight_distribution.csv",
    output_png="plots/layerwise_std_ratio.png",
):
    """
    Similar to plot_layerwise_l2norm_ratio, but uses the 'std' column instead.
    We compute ratio = (std of pruned) / (std of raw), per layer.
    This can reveal if certain layers' standard deviation is reduced more
    strongly for certain subtask prunings, even though the fraction is uniform.

    The logic is parallel: we match each pruned model to its raw baseline
    by name, then gather per-layer std, do ratio, line-plot vs. layer index.
    """
    import csv
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np

    def extract_layer_index(layer_str):
        import re

        match = re.search(r"\.layers\.(\d+)\.", layer_str)
        if match:
            return int(match.group(1))
        else:
            return None

    all_data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse numeric
            try:
                row["std"] = float(row["std"])
            except:
                row["std"] = 0
            row["layer_index"] = extract_layer_index(row["layer"])
            all_data.append(row)

    from collections import defaultdict

    model_to_layer_std = defaultdict(dict)

    def make_model_key(r):
        return (r["type"], r["name"], r["pruned_on"], r["sparsity"])

    for r in all_data:
        mk = make_model_key(r)
        idx = r["layer_index"]
        if idx is not None:
            model_to_layer_std[mk][idx] = r["std"]

    # find raw baselines by name
    raw_models_by_name = {}
    for mk in model_to_layer_std.keys():
        (mtype, mname, pon, sp) = mk
        if mtype == "raw":
            raw_models_by_name[mname] = mk

    # build ratio data
    ratio_data = defaultdict(dict)

    for mk in model_to_layer_std.keys():
        (mtype, mname, pon, sp) = mk
        if mtype == "pruned":
            if mname in raw_models_by_name:
                raw_mk = raw_models_by_name[mname]
                std_pruned = model_to_layer_std[mk]
                std_raw = model_to_layer_std[raw_mk]
                # union of idx
                all_idx = sorted(set(std_pruned.keys()) & set(std_raw.keys()))
                for idx in all_idx:
                    if std_raw[idx] != 0:
                        ratio = std_pruned[idx] / std_raw[idx]
                    else:
                        ratio = 0
                    ratio_data[mk][idx] = ratio

    # gather lines
    def parse_sparsity(sp):
        try:
            return float(sp.replace("%", ""))
        except:
            return 0

    curves = []
    for mk in ratio_data.keys():
        (mtype, mname, pon, sp) = mk
        if pon in (None, ""):
            continue
        label_str = f"{pon}({sp}) {mname}"
        layer_idx_sorted = sorted(ratio_data[mk].keys())
        yvals = [ratio_data[mk][i] for i in layer_idx_sorted]
        curves.append(
            (pon, parse_sparsity(sp), mname, layer_idx_sorted, yvals, label_str)
        )

    curves.sort(key=lambda x: (x[0], x[1], x[2]))

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in curves:
        _, _, _, layer_idx_sorted, yvals, label_str = c
        ax.plot(layer_idx_sorted, yvals, marker="o", label=label_str, alpha=0.8)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Std (pruned) / Std (raw)")
    ax.set_title("Layerwise Std Ratio (pruned ÷ raw)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved std ratio plot to {output_png}")


###############################################################################
# MAIN EXAMPLE
###############################################################################

if __name__ == "__main__":
    # Example usage:
    plot_tom_eval(
        csv_file="results/tom_eval.csv",
        output_png="plots/Effect of Subtask-Specific Pruning on ToM Performance.png",
    )

    plot_eval_zero_shot_summary(
        csv_file="results/eval_zero_shot_summary.csv",
        output_png="plots/Zero-Shot Benchmark Accuracy and Language Model Perplexity.png",
    )

    """plot_layerwise_weight_distribution_1(
        "results/layerwise_weight_distribution.csv", "plots/layerwise_dist_summary.png"
    )
    plot_layerwise_weight_distribution_2(
        "results/layerwise_weight_distribution.csv", "plots/layerwise_dist_boxplot.png"
    )"""

    """plot_layerwise_l2norm_ratio()
    plot_layerwise_std_ratio()"""
    print("Done generating plots!")
