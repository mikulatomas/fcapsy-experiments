from itertools import combinations
import os

from fcapsy import Context, Concept
from fcapsy.psychology.typicality import typicality_avg
from fcapsy.similarity import smc, jaccard, rosch
from fcapsy.utils import iterator_mean
import pandas as pd
import plotly.express as px

from fcapsy_experiments.utils import text_to_filename, fig_to_file


def k_values_or_until_differs(iterator, k):
    max_idx = len(iterator) - 1

    if k == 0:
        return []

    for idx in range(len(iterator)):
        if idx + 1 < k or (idx < max_idx and iterator[idx] == iterator[idx + 1]):
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


def exp_top_k_similarity(concept,
                         context,
                         k_range=None,
                         ground_truth=None,
                         typicality_metrics={
                             typicality_avg: [smc, jaccard, rosch]}):

    if k_range is None:
        k_range = range(1, len(concept.extent))

    results = []

    for typicality, similarity_functions in typicality_metrics.items():
        for sim1, sim2 in combinations(similarity_functions, 2):
            sim1_order = sorted(concept.extent,
                                key=lambda x: typicality(
                                    x, concept, context, sim1),
                                reverse=True)

            sim2_order = sorted(concept.extent,
                                key=lambda x: typicality(
                                    x, concept, context, sim2),
                                reverse=True)

            label = f"{typicality.short_name} {sim1.short_name}-{sim2.short_name}"

            for k in k_range:
                results.append([k, top_k_similarity(
                    sim1_order, sim2_order, k, jaccard, context), label])

    # Add ground truth comparation
    if ground_truth is not None:
        for typicality, similarity_functions in typicality_metrics.items():
            for sim in similarity_functions:
                sim_order = sorted(concept.extent,
                                   key=lambda x: typicality(
                                       x, concept, context, sim),
                                   reverse=True)

                gt_order = list(ground_truth.sort_values(
                    ground_truth.columns[0], ascending=False).index)

                label = f"{typicality.short_name} {sim.short_name}-GT"

                for k in k_range:
                    results.append([k, top_k_similarity(
                        sim_order, gt_order, k, jaccard, context), label])

    return pd.DataFrame(results, columns=['k', 'top_k_similarity', 'label'])


# def experiment_top_k_similarity(dataset,
#                                 extent=None,
#                                 intent=None,
#                                 k_range=None,
#                                 ground_truth=None,
#                                 typicality_metrics={
#                                     typicality_avg: [smc, jaccard, rosch]}):
#     context = Context.from_pandas(dataset)

#     if extent is not None:
#         concept = Concept.from_extent(extent, context)
#         if set(concept.extent) != set(extent):
#             raise ValueError("Extent does not form formal concept!")
#     elif intent is not None:
#         concept = Concept.from_intent(intent, context)
#         if set(concept.intent) != set(intent):
#             raise ValueError("Intent does not form formal concept!")
#     else:
#         raise ValueError('Extent or Intent must be provided!')

#     if k_range is None:
#         k_range = range(1, len(concept.extent))

#     results = []

#     for typicality, similarity_functions in typicality_metrics.items():
#         for sim1, sim2 in combinations(similarity_functions, 2):
#             sim1_order = sorted(concept.extent,
#                                 key=lambda x: typicality(
#                                     x, concept, context, sim1),
#                                 reverse=True)

#             sim2_order = sorted(concept.extent,
#                                 key=lambda x: typicality(
#                                     x, concept, context, sim2),
#                                 reverse=True)

#             label = f"{typicality.short_name} {sim1.short_name}-{sim2.short_name}"

#             for k in k_range:
#                 results.append([k, top_k_similarity(
#                     sim1_order, sim2_order, k, jaccard, context), label])

#     # Add ground truth comparation
#     if ground_truth is not None:
#         for typicality, similarity_functions in typicality_metrics.items():
#             for sim in similarity_functions:
#                 sim_order = sorted(concept.extent,
#                                    key=lambda x: typicality(
#                                        x, concept, context, sim),
#                                    reverse=True)

#                 gt_order = list(ground_truth.sort_values(
#                     ground_truth.columns[0], ascending=False).index)

#                 label = f"{typicality.short_name} {sim.short_name}-GT"

#                 for k in k_range:
#                     results.append([k, top_k_similarity(
#                         sim_order, gt_order, k, jaccard, context), label])

#     return pd.DataFrame(results, columns=['k', 'top_k_similarity', 'label'])


def plot_top_k_similarity(df, title, output_dir=None):
    fig = px.line(df, x='k', y='top_k_similarity', color='label', labels={
        'label': 'Legend'}, line_dash='label', title=title)

    if output_dir:
        output_filename = f"{plot_top_k_similarity.__name__}_{text_to_filename(title)}"

        fig_to_file(fig, output_dir, output_filename)
    return fig
