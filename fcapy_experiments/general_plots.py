import plotly.figure_factory as ff
from fcapy_experiments.utils import fig_to_file, text_to_filename


def plot_table(df, title, index_title="", decimals=3, width=None, output_dir=None, output_filename=None):
    if width is None:
        width = len(df.columns) * 120

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
        if output_filename is None:
            output_filename = text_to_filename(title)

        fig_to_file(fig, output_dir, output_filename)

    return fig
