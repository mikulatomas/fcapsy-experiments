import itertools

import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from fcapsy.typicality import typicality_avg, _get_vectors
from fcapsy.similarity import jaccard, smc, rosch

from fcapsy_experiments._styles import css, css_typ


class ConceptTypicality:
    def __init__(
        self, concept, axis=0, human_judgements=None, count=False, extra_columns=None, typicality_functions=None
    ) -> None:
        if typicality_functions is None:
            typicality_functions = {
                "typ_avg": {
                    "func": typicality_avg,
                    "args": {"J": [jaccard], "SMC": [smc], "R": [rosch]},
                }
            }

        self.df = self._init(concept, axis, human_judgements, count, typicality_functions, extra_columns)

        if extra_columns:
            extra_columns = list(extra_columns.keys())
        
        self.extra_columns = extra_columns

    @staticmethod
    def _init(concept, axis, human_judgements, count, typicality_functions, extra_columns):
        items = concept.extent

        if axis == 1:
            items = concept.intent

        columns = []

        for name, typicality in typicality_functions.items():
            if typicality["args"]:
                for arg in typicality["args"].keys():
                    columns.append(f"{name}({arg})")
            else:
                columns.append(f"{name}")

        df = pd.DataFrame(index=items, columns=columns, dtype=float)

        for item in items:
            row = []

            for typicality in typicality_functions.values():
                function = typicality["func"]
                args = typicality["args"].values()
                if args:
                    row.extend([function(item, concept, *arg) for arg in args])
                else:
                    row.append(function(item, concept))

            df.loc[item] = row

        if human_judgements is not None:
            df["HJ"] = human_judgements
        
        if count:
            context = concept.lattice._context

            if axis == 0:
                names = context.objects
                counts = (intent.bits().count("1") for intent in context._intents)
            else:
                names = context.properties
                counts = (extent.bits().count("1") for extent in context._extents)
            
            df["Count"] = [row[1] for row in filter(lambda x: x[0] in items, zip(names, counts))]
        
        if extra_columns:
            for name, values in extra_columns.items():
                df[name] = values

        return df

    def to_html(self):
        final_table = pd.DataFrame()

        for column in self.df.columns:
            round_and_sort = self.df.sort_values(column, ascending=False, kind="mergesort")

            final_table[f"{column} order"] = round_and_sort.index
            final_table[column] = round_and_sort.reset_index()[column]

        df = final_table.reset_index().drop("index", axis=1)

        df = df.style.format(precision=3)
        df.background_gradient(cmap="RdYlGn")
        df.set_table_styles(css + css_typ)
        df.hide_index()

        return df.to_html()

    def to_plotly(self):
        markers = ["square", "diamond", "triangle-up", "circle", "pentagon"]

        scatters = []

        df = self.df.sort_values(self.df.columns[0], ascending=False, kind="mergesort")

        if "Count" in df.columns:
            # scaling Count column 
            scaler = MinMaxScaler()
            df["Count"] = scaler.fit_transform(df["Count"].values.reshape(-1, 1))
        
        if self.extra_columns:
            for extra in self.extra_columns:
                scaler = MinMaxScaler()
                df[extra] = scaler.fit_transform(df[extra].values.reshape(-1, 1))

        for column, marker in zip(df.columns, itertools.cycle(markers)):
            scatters.append(
                go.Scatter(
                    name=column,
                    x=df.index,
                    y=df[column],
                    mode="markers",
                    marker_symbol=marker,
                    marker_line_width=1,
                    marker_size=8,
                )
            )

        fig = go.Figure(data=scatters)

        fig.update_layout(
            font_family="IBM Plex Sans",
            font_color="black",
            margin=dict(l=15, r=15, t=15, b=15),
            xaxis=dict(
                title="objects",
                mirror=True,
                ticks="inside",
                showline=True,
                linecolor="black",
                linewidth=1,
                tickangle=-90,
                gridcolor="rgba(200,200,200,1)",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="typicality",
                mirror=True,
                ticks="inside",
                showline=True,
                linecolor="black",
                linewidth=1,
                range=[0, 1],
                dtick=0.1,
                gridcolor="rgba(200,200,200,0)",
                tickfont=dict(size=11),
            ),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                # yanchor="bottom",
                # y=0.05,
                # xanchor="right",
                # x=0.97,
                font=dict(size=10),
            ),
        )

        return fig

    def to_plotly_html(self):
        return self.to_plotly().to_html(
            full_html=False,
            include_plotlyjs="cdn",
            include_mathjax="cdn",
            default_width=700,
            default_height=390
        )
