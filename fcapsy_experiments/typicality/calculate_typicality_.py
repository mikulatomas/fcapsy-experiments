import os

import numpy as np

from fcapsy import Context, Concept
from fcapsy.psychology.typicality import typicality_avg
from fcapsy.similarity import smc, jaccard, rosch
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from fcapsy_experiments.utils import text_to_filename, fig_to_file
from fcapsy_experiments.general_plots import plot_table

from fcapsy.algorithms.lindig import upper_neighbors, lower_neighbors


def exp_concept_typicality(concept,
                           context,
                           ground_truths=None,
                           axis=0,
                           typicality_metrics={
                               typicality_avg: [smc, jaccard, rosch]}):
    columns = []

    for typicality, similarity_functions in typicality_metrics.items():
        try:
            typ_name = typicality.short_name
        except:
            typ_name = 'N/A'

        for similarity_function in similarity_functions:
            try:
                sim_name = similarity_function.short_name
            except:
                sim_name = 'N/A'

            columns.append(f"{typ_name}({sim_name})")

    item_set = concept.extent

    if axis == 1:
        item_set = concept.intent

    df = pd.DataFrame(
        columns=columns,
        index=item_set.members(),
        dtype=float)

    for item in item_set:
        row = []

        for typicality, similarity_functions in typicality_metrics.items():
            row.extend([typicality(item, concept, context, similarity_function, axis=axis)
                        for similarity_function in similarity_functions])

        df.loc[item] = row

    if ground_truths is not None:
        for name, values in ground_truths.items():
            before_join = set(df.columns)

            df = df.join(values[values.columns[0]])

            human_rating_column = list(
                before_join.symmetric_difference(set(df.columns)))[0]

            df = df.rename(columns={human_rating_column: name})

    return df


def exp_mean_typicality_across_concepts(concepts,
                                        context,
                                        typicality_metrics={
                                            typicality_avg: [smc, jaccard, rosch]},
                                        ):
    rows = []
    for concept in concepts:
        typicalities = exp_concept_typicality(concept, context)
        mean = typicalities.mean()

        rows.append(mean)

    df = pd.DataFrame(rows, columns=typicalities.columns)

    return df.mean()


def exp_mean_typicality_neighbor_concepts(concept,
                                          context,
                                          typicality_metrics={
                                              typicality_avg: [smc, jaccard, rosch]}):

    typicalities = exp_concept_typicality(concept, context)

    mean_typ = typicalities.mean()

    upper_neighbor = upper_neighbors(context, concept)
    lower_neighbor = lower_neighbors(context, concept)

    mean_upper_typ = exp_mean_typicality_across_concepts(
        upper_neighbor, context, typicality_metrics=typicality_metrics)

    mean_lower_typ = exp_mean_typicality_across_concepts(
        lower_neighbor, context, typicality_metrics=typicality_metrics)

    index = [f'{concept.name} upper', concept.name, f'{concept.name} lower']
    rows = [mean_upper_typ, mean_typ, mean_lower_typ]

    return pd.DataFrame(rows, index=index, columns=typicalities.columns)


def exp_corr_across_concepts(concepts,
                             context,
                             corr_method,
                             corr_to,
                             ground_truths=None,
                             typicality_metrics={
                                 typicality_avg: [smc, jaccard, rosch]},
                             ):

    typicalities = [exp_concept_typicality(
        concept, context, gt, typicality_metrics) for concept, gt in zip(concepts, ground_truths)]

    correlations = []
    correlations_index = []

    for concept, typicality in zip(concepts, typicalities):
        correlation = typicality.corr(method=corr_method)
        correlation = correlation.drop(corr_to, axis=1)

        correlations.append(correlation.loc[corr_to])
        correlations_index.append(concept.name)

    return pd.DataFrame(correlations,
                        index=correlations_index, columns=correlation.columns)


def exp_weighted_mean_corr_across_concepts(concepts,
                                           context,
                                           corr_method,
                                           corr_to,
                                           ground_truths=None,
                                           typicality_metrics={
                                               typicality_avg: [smc, jaccard, rosch]},
                                           ):

    typicalities = [exp_concept_typicality(
        concept, context, gt, typicality_metrics) for concept, gt in zip(concepts, ground_truths)]

    correlations = []

    for _, typicality in zip(concepts, typicalities):
        correlation = typicality.corr(method=corr_method)[
            corr_to].drop([corr_to])
        sample_size = typicality.shape[0]

        correlations.append(list(correlation.values) + [sample_size])

    df = pd.DataFrame(correlations, columns=list(
        correlation.index) + ['Sample Size'])

    # calculate weighted mean
    for column in df.columns[:-1]:
        df[column] = df[column] * df['Sample Size']

    avg = df.sum()
    for column in avg.index[:-1]:
        avg[column] = avg[column] / avg['Sample Size']

    return avg.drop('Sample Size')


def exp_mean_corr_across_concepts(concepts,
                                  context,
                                  corr_method,
                                  corr_to,
                                  ground_truths=None,
                                  typicality_metrics={
                                      typicality_avg: [smc, jaccard, rosch]},
                                  ):

    typicalities = [exp_concept_typicality(
        concept, context, gt, typicality_metrics) for concept, gt in zip(concepts, ground_truths)]

    correlations = []

    for _, typicality in zip(concepts, typicalities):
        correlation = typicality.corr(method=corr_method)[
            corr_to].drop([corr_to])

        correlations.append(correlation.values)

    df = pd.DataFrame(correlations, columns=correlation.index)

    return df.mean()


def create_typicality_table(df, decimals=3):
    final_table = pd.DataFrame()

    for column in df.columns:
        round_and_sort = df.round(decimals=decimals).sort_values(
            column, ascending=False)

        final_table[f'{column} order'] = round_and_sort.index
        final_table[column] = round_and_sort.reset_index()[column]

    final_table = final_table.reset_index().drop('index', axis=1)

    return final_table


def plot_typicality(df, title, decimals=3, width=None, output_dir=None):
    final_table = create_typicality_table(df, decimals=decimals)

    if width is None:
        width = len(df.columns) * 230

    fig = ff.create_table(final_table)
    fig.layout.width = width
    fig.update_layout(
        title={
            'text': title,
            'x': 0.01,
            'xanchor': 'left'},
        margin={'t': 70})

    fig.layout.width = width

    if output_dir:
        output_filename = f"{plot_typicality.__name__}_{text_to_filename(title)}"

        fig_to_file(fig, output_dir, output_filename)

    return fig


def plot_mean_typicality_neighbor_concepts(df, title, index_title="", decimals=3, width=None, output_dir=None):
    if width is None:
        width = len(df.columns) * 200

    if output_dir:
        output_filename = f"{plot_mean_typicality_neighbor_concepts.__name__}_{text_to_filename(title)}"

        if index_title != "":
            output_filename = f"{output_filename}_{index_title}"
    else:
        output_filename = None

    return plot_table(df, title, index_title=index_title, decimals=decimals, width=width, output_dir=output_dir, output_filename=output_filename)


def plot_typicality_correlation(df, title, index_title="", decimals=3, width=None, output_dir=None):
    if width is None:
        width = len(df.columns) * 200

    if output_dir:
        output_filename = f"{plot_typicality_correlation.__name__}_{text_to_filename(title)}"

        if index_title != "":
            output_filename = f"{output_filename}_{index_title}"
    else:
        output_filename = None

    return plot_table(df, title, index_title=index_title, decimals=decimals, width=width, output_dir=output_dir, output_filename=output_filename)


def plot_boxplot_typicality(df, title, output_dir=None):
    fig = px.box(df, y=df.columns, title=title)

    if output_dir:
        output_filename = f"{plot_boxplot_typicality.__name__}_{text_to_filename(title)}"

        fig_to_file(fig, output_dir, output_filename)

    return fig
