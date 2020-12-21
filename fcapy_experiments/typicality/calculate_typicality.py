from fcapy import Context, Concept
from fcapy.psychology.typicality import typicality_avg
from fcapy.similarity.objects import smc, jaccard, rosch
import pandas as pd
import plotly.figure_factory as ff
from fcapy_experiments.utils import text_to_filename

import os


def experiment_calculate_typicality(dataset,
                                    extent=None,
                                    intent=None,
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

    df = pd.DataFrame(
        columns=[
            similarity_function.short_name for similarity_function in similarity_functions],
        index=concept.extent.members(),
        dtype=float)

    rows = tuple(context.filter(concept.extent))

    for obj, item in zip(concept.extent, rows):
        df.loc[obj] = [typicality(item, rows, similarity_function)
                       for similarity_function in similarity_functions]

    if ground_truth is not None:
        before_join = set(df.columns)

        df = df.join(ground_truth[ground_truth.columns[0]])

        human_rating_column = list(
            before_join.symmetric_difference(set(df.columns)))[0]

        df = df.rename(columns={human_rating_column: "GT"})

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
        width = len(df.columns) * 250

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
        fig.write_image(os.path.join(
            output_dir, f"{plot_calculate_typicality.__name__}_{text_to_filename(title)}.pdf"))

    return fig
