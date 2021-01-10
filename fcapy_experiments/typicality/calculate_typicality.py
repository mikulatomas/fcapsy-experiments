import os

from fcapy import Context, Concept
from fcapy.psychology.typicality import typicality_avg
from fcapy.similarity.objects import smc, jaccard, rosch
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from fcapy_experiments.utils import text_to_filename, fig_to_file


def experiment_calculate_typicality(dataset,
                                    extent=None,
                                    intent=None,
                                    ground_truths=None,
                                    typicality_metrics={
                                        typicality_avg: [smc, jaccard, rosch]}):
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

    columns = []

    for typicality, similarity_functions in typicality_metrics.items():
        columns.extend([
            f"{typicality.short_name}({similarity_function.short_name})" for similarity_function in similarity_functions])

    df = pd.DataFrame(
        columns=columns,
        index=concept.extent.members(),
        dtype=float)

    for obj in concept.extent:
        row = []

        for typicality, similarity_functions in typicality_metrics.items():
            row.extend([typicality(obj, concept, context, similarity_function)
                        for similarity_function in similarity_functions])

        df.loc[obj] = row

    if ground_truths is not None:
        for name, values in ground_truths.items():
            before_join = set(df.columns)

            df = df.join(values[values.columns[0]])

            human_rating_column = list(
                before_join.symmetric_difference(set(df.columns)))[0]

            df = df.rename(columns={human_rating_column: name})

    return df


def plot_calculate_typicality(df, title, decimals=3, width=None, output_dir=None):
    final_table = pd.DataFrame()

    for column in df.columns:
        round_and_sort = df.round(decimals=decimals).sort_values(
            column, ascending=False)

        final_table[f'{column} order'] = round_and_sort.index
        final_table[column] = round_and_sort.reset_index()[column]

    final_table = final_table.reset_index().drop('index', axis=1)

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
        output_filename = f"{plot_calculate_typicality.__name__}_{text_to_filename(title)}"

        fig_to_file(fig, output_dir, output_filename)

    return fig


def plot_typicality_correlation(df, title, index_title="", decimals=3, width=None, output_dir=None):
    if width is None:
        width = len(df.columns) * 150

    df = df.round(decimals=decimals)

    fig = ff.create_table(df, index=True, index_title=index_title)
    fig.layout.width = width
    fig.update_layout(
        title={
            'text': title,
            'x': 0.01,
            'xanchor': 'left'},
        margin={'t': 70})

    fig.layout.width = width

    if output_dir:
        output_filename = f"{plot_typicality_correlation.__name__}_{text_to_filename(title)}"

        if index_title != "":
            output_filename = f"{output_filename}_{index_title}"

        fig_to_file(fig, output_dir, output_filename)

    return fig


def plot_boxplot_typicality(df, title, output_dir=None):
    fig = px.box(df, y=df.columns, title=title)

    if output_dir:
        output_filename = f"{plot_boxplot_typicality.__name__}_{text_to_filename(title)}"

        fig_to_file(fig, output_dir, output_filename)

    return fig
