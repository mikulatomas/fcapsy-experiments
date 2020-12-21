from fcapy import Context, Concept
from fcapy.psychology.typicality import typicality_avg
from fcapy.similarity.objects import smc, jaccard, rosch
from fcapy.utils import iterator_mean
from fcapy_experiments.utils import text_to_filename
import pandas as pd
import plotly.express as px

from itertools import combinations
import os


def k_values_or_until_differs(iterator, k):
    max_idx = len(iterator) - 1

    for idx in range(len(iterator)):
        if idx < k or (idx < max_idx and iterator[idx] == iterator[idx + 1]):
            yield iterator[idx]
        else:
            yield iterator[idx]
            break


def top_k_similarity(metric1_order, metric_2_order, k, similarity_function, context):
    top_k_objects1 = list(k_values_or_until_differs(metric1_order, k))
    top_k_objects2 = list(k_values_or_until_differs(metric_2_order, k))

    i1 = iterator_mean(
        [max(
            [similarity_function(b1, b2) for b2 in context.filter(top_k_objects1)]) for b1 in context.filter(top_k_objects2)])

    i2 = iterator_mean(
        [max(
            [similarity_function(b1, b2) for b2 in context.filter(top_k_objects2)]) for b1 in context.filter(top_k_objects1)])

    return min(i1, i2)


def experiment_top_k_similarity(dataset,
                                extent=None,
                                intent=None,
                                k_range=None,
                                ground_truth=None,
                                typicality=typicality_avg,
                                similarity_functions=[smc, jaccard, rosch]):
    context = Context.from_pandas(dataset)

    if extent is not None:
        concept = Concept.from_extent(extent, context)
        if set(concept.extent) != set(extent):
            raise ValueError("Extent does not form formal concept!")
    elif intent is not None:
        concept = Concept.from_intent(intent, context)
        if set(concept.intent) != set(intent):
            raise ValueError("Intent does not form formal concept!")
    else:
        raise ValueError('Extent or Intent must be provided!')

    if k_range is None:
        k_range = range(len(concept.extent))

    results = []

    for sim1, sim2 in combinations(similarity_functions, 2):
        sim1_order = sorted(concept.extent,
                            key=lambda x: typicality_avg(
                                next(context.filter([x])), context.filter(concept.extent), sim1),
                            reverse=True)

        sim2_order = sorted(concept.extent,
                            key=lambda x: typicality_avg(
                                next(context.filter([x])), context.filter(concept.extent), sim2),
                            reverse=True)

        label = f"{sim1.short_name}-{sim2.short_name}"

        for k in k_range:
            results.append([k, top_k_similarity(
                sim1_order, sim2_order, k, smc, context), label])

    # Add ground truth comparation
    if ground_truth is not None:
        for sim in similarity_functions:
            sim_order = sorted(concept.extent,
                               key=lambda x: typicality_avg(
                                   next(context.filter([x])), context.filter(concept.extent), sim),
                               reverse=True)

            gt_order = list(ground_truth.sort_values(
                ground_truth.columns[0], ascending=False).index)

            label = f"GT-{sim.short_name}"

            for k in k_range:
                results.append([k, top_k_similarity(
                    sim_order, gt_order, k, smc, context), label])

    return pd.DataFrame(results, columns=['k', 'top_k_similarity', 'label'])


def plot_top_k_similarity(df, title, output_dir=None):
    fig = px.line(df, x='k', y='top_k_similarity', color='label', labels={
        'label': 'Legend'}, line_dash='label', title=title)

    if output_dir:
        fig.write_image(os.path.join(
            output_dir, f"{plot_top_k_similarity.__name__}_{text_to_filename(title)}.pdf"))

    return fig
