import typing

import pandas as pd
import plotly.express as px


class TopRIntersection:
    def __init__(
        self,
        source: "pd.DataFrame",
        context: "concepts.Context",
        to_columns: typing.Optional[typing.List[str]] = None,
        specific_r: typing.Optional[int] = None,
        percentage: bool = False
    ) -> None:
        if to_columns is None:
            to_columns = source.columns

        self._source = source
        self._context = context
        self.percentage = percentage
        self.specific_r = specific_r
        self.df = self._init(self, to_columns, self.percentage, self.specific_r)

    @staticmethod
    def _get_column_orders(df, tuples):
        results = []
        labels = []
        a = []
        b = []

        for column1, column2 in tuples:
            column1_order = list(
                df.sort_values(column1, ascending=False, kind="mergesort").index
            )

            column2_order = list(
                df.sort_values(column2, ascending=False, kind="mergesort").index
            )

            results.append([column1_order, column2_order])
            labels.append(f"{column1}-{column2}")
            a.append(column1)
            b.append(column2)

        return results, labels, a, b

    @staticmethod
    def _top_r_intersection(metric_1_order, metric_2_order, r):
        return len(set(metric_1_order[:r]).intersection(set(metric_2_order[:r])))

    @staticmethod
    def _init(inst, to_columns, percentage, specific_r):
        r_range = range(1, len(inst._source.index) + 1)

        if specific_r:
            r_range = range(specific_r, specific_r + 1)

        results = []

        columns_tuples, labels, a, b = inst._get_column_orders(
            inst._source,
            [(x, y) for x in inst._source.columns for y in to_columns if x != y],
        )

        for (column1_order, column2_order), label, a_, b_ in zip(columns_tuples, labels, a, b):
            for r in r_range:
                intersection_size = inst._top_r_intersection(column1_order, column2_order, r)

                if percentage:
                    intersection_size = intersection_size / r

                results.append(
                    [
                        r,
                        intersection_size,
                        label,
                        a_,
                        b_,
                    ]
                )

        return pd.DataFrame(
            results, columns=["r", "top_r_intersection", "label", "a", "b"]
        )

    def to_plotly(self) -> "go.Figure":
        """Generates plotly figure.

        Returns:
            go.Figure: figure
        """
        fig = px.line(
            self.df,
            x="r",
            y=self.df.columns[1],
            color="label",
            labels={"label": "Legend"},
            line_dash="label",
        )

        # layout needs some cleaning
        fig.update_layout(
            title=dict(
                font=dict(family="IBM Plex Sans", size=14, color="black"),
            ),
            margin=dict(l=15, r=15, t=15, b=15),
            xaxis=dict(
                title="r",
                mirror=True,
                ticks="inside",
                showline=True,
                linecolor="black",
                linewidth=1,
                tickangle=-90,
                tickfont=dict(family="IBM Plex Sans", size=11, color="black"),
            ),
            yaxis=dict(
                title="S",
                mirror=True,
                ticks="inside",
                # range=[0, 1],
                showline=True,
                linecolor="black",
                linewidth=1,
                tickfont=dict(family="IBM Plex Sans", size=11, color="black"),
            ),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                font=dict(family="IBM Plex Sans", size=10),
            ),
        )

        return fig

    def to_plotly_html(
        self, default_width: int = 700, default_height: int = 390
    ) -> str:
        """Generates html version of plotly graph

        Args:
            default_width (int, optional): default graph width. Defaults to 700.
            default_height (int, optional): default graph height. Defaults to 390.

        Returns:
            str: graph html
        """
        return self.to_plotly().to_html(
            full_html=False,
            include_plotlyjs="cdn",
            include_mathjax="cdn",
            default_width=default_width,
            default_height=default_height,
        )
