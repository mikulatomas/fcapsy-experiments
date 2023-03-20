import plotly.graph_objects as go

from plotly.subplots import make_subplots


def plot_respondents_histogram(df, title, columns=5):
    drop_columns = len(df.columns) - len(list(filter(lambda x: "respondent" in x, df.columns)))
    
    df = df.drop(df.columns[-drop_columns:], axis=1).T
    max_value = df.shape[0]

    rows = int(len(df.columns) / columns) + 1

    fig = make_subplots(rows=rows, cols=columns, subplot_titles=df.columns)

    for idx, col in enumerate(df.columns):
        col_idx = (idx % columns) + 1
        row_idx = (idx // columns) + 1

        fig.add_trace(
            go.Histogram(x=df[col]),
            row=row_idx, col=col_idx,
        )

    fig.update_yaxes(range=[0, max_value])
    fig.update_layout(height=1000, width=800, title_text=title, showlegend=False)

    return fig