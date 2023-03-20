import typing
import operator

from statistics import mean
from itertools import combinations

import pandas as pd
import plotly.express as px


class ROrderComparison:
    """_summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_

    Example:
        >>> import pandas as pd
        >>> from binsdpy import similarity
        >>> from concepts import Context
        >>> from fcapsy_experiments.typicality import ROrderComparison
        >>> objects = ["a", "b", "c", "d", "e"]
        >>> attributes = ["a1", "a2"]
        >>> bools = [[True, False], [True, True], [True, False], [True, False], [True, True]]
        >>> context = Context(objects, attributes, bools)
        >>> o1 = [5, 4, 3, 2, 1]
        >>> o2 = [4, 5, 1, 2, 3]
        >>> df = pd.DataFrame(zip(o1, o2), columns = ["o1", "o2"], index=objects)
        >>> df
           o1  o2
        a   5   4
        b   4   5
        c   3   1
        d   2   2
        e   1   3
        >>> order_comparison = ROrderComparison(df, context, specific_r=2)
        >>> order_comparison.df
           r  r_order_comparison  label   a   b
        0  2                   2  o1-o2  o1  o2
        >>> order_comparison = ROrderComparison(df, context, mode="bottom", specific_r=2, percentage=True)
        >>> order_comparison.df
           r  r_order_comparison  label   a   b
        0  2                 0.5  o1-o2  o1  o2
        >>> order_comparison = ROrderComparison(df, context, mode="topbottom", specific_r=2, percentage=True)
        >>> order_comparison.df
           r  r_order_comparison  label   a   b
        0  2                0.75  o1-o2  o1  o2
        >>> order_comparison = ROrderComparison(df, context, similarity=similarity.jaccard, specific_r=2)
        >>> order_comparison.df
           r  r_order_comparison  label   a   b
        0  2                 1.0  o1-o2  o1  o2
        >>> order_comparison = ROrderComparison(df, context, mode="bottom", similarity=similarity.jaccard, specific_r=2)
        >>> order_comparison.df
           r  r_order_comparison  label   a   b
        0  2                0.75  o1-o2  o1  o2
    """

    supported_modes = ["topbottom", "top", "bottom"]

    def __init__(
        self,
        source: "pd.DataFrame",
        context: "concepts.Context",
        mode: str = "top",
        percentage: bool = False,
        to_columns: typing.Optional[typing.List[str]] = None,
        similarity: typing.Optional[typing.Callable] = None,
        specific_r: typing.Optional[int] = None,
    ):
        if to_columns is None:
            to_columns = list(source.columns)

        if mode not in self.supported_modes:
            raise ValueError(
                f"Mode {mode} is not supported, only {self.supported_modes}"
            )

        if percentage and similarity is not None:
            raise ValueError(
                f"Similarity and percentage are not avaliable combination."
            )

        if specific_r:
            if specific_r > len(source.index):
                raise ValueError(
                    f"Specific r ({specific_r}) is larger than maximum number of items in given source ({len(source.index)})."
                )

        self.mode = mode
        self.percentage = percentage
        self.similarity = similarity
        self.specific_r = specific_r
        self.to_columns = to_columns

        self._source = source
        self._context = context

        self.df = self._init()

    def _init(self):
        r_range = range(1, len(self._source.index) + 1)
        # r_range = range(len(self._source.index) + 1)

        if self.specific_r:
            r_range = range(self.specific_r, self.specific_r + 1)

        results = []

        columns_tuples, labels, a, b = self._get_column_orders(
            self._source,
            list(combinations(self._source.columns, r=2))
            # [(x, y) for x in self._source.columns for y in self.to_columns if x != y],
        )

        if self.similarity is None:
            r_operation = self._r_intersection
            aggregation_op = operator.add
            # scale_by = 2
        else:
            r_operation = self._r_similarity
            aggregation_op = operator.mul
            # scale_by = 1

        for (column1_order, column2_order), label, a_, b_ in zip(
            columns_tuples, labels, a, b
        ):
            for r in r_range:
                if self.mode == "top":
                    calculated_value = r_operation(r, column1_order, column2_order)
                elif self.mode == "bottom":
                    column1_order_reversed = column1_order[::-1]
                    column2_order_reversed = column2_order[::-1]
                    calculated_value = r_operation(
                        r,
                        column1_order_reversed,
                        column2_order_reversed,
                    )
                else:
                    # TODO - test scaling
                    # calculated_value = aggregation_op(top_r, bottom_r) / scale_by
                    column1_order_reversed = column1_order[::-1]
                    column2_order_reversed = column2_order[::-1]
                    top_r = r_operation(r, column1_order, column2_order)
                    bottom_r = r_operation(
                        r,
                        column1_order_reversed,
                        column2_order_reversed,
                    )
                    calculated_value = aggregation_op(top_r, bottom_r)

                if self.percentage:
                    # if r != 0:
                    if self.mode in ["top", "bottom"]:
                        calculated_value = calculated_value / r
                    elif self.mode == "topbottom":
                        calculated_value = calculated_value / (2 * r)

                    # r = r / len(self._source.index)

                results.append(
                    [
                        r,
                        calculated_value,
                        label,
                        a_,
                        b_,
                    ]
                )

        return pd.DataFrame(
            results, columns=["r", "r_order_comparison", "label", "a", "b"]
        )

    def _r_similarity(self, r, metric_1_order, metric_2_order):
        def _get_vectors(context, items):
            try:
                label_domain = context._extents
                vectors = tuple(
                    map(label_domain.__getitem__, map(context.properties.index, items))
                )
            except ValueError:
                label_domain = context._intents
                vectors = tuple(
                    map(label_domain.__getitem__, map(context.objects.index, items))
                )
            return vectors

        v1 = list(self._r_values_or_until_differs(metric_1_order, r))
        v2 = list(self._r_values_or_until_differs(metric_2_order, r))

        if not v1 and not v2:
            return 0

        vectors_1 = _get_vectors(self._context, v1)

        vectors_2 = _get_vectors(self._context, v2)

        i1 = mean(
            (max((self.similarity(b1, b2) for b2 in vectors_1)) for b1 in vectors_2)
        )

        i2 = mean(
            (max((self.similarity(b1, b2) for b2 in vectors_2)) for b1 in vectors_1)
        )

        return min(i1, i2)

    def _r_intersection(self, r, metric_1_order, metric_2_order):
        # print(r)
        # print(metric_1_order[:r])
        # print(metric_2_order[:r])
        return len(set(metric_1_order[:r]).intersection(set(metric_2_order[:r])))

    @staticmethod
    def _r_values_or_until_differs(iterator, r):
        max_idx = len(iterator) - 1

        if r == 0:
            return []

        for idx in range(len(iterator)):
            if idx + 1 < r or (idx < max_idx and iterator[idx] == iterator[idx + 1]):
                yield iterator[idx]
            else:
                yield iterator[idx]
                break

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

    def to_plotly(self) -> "go.Figure":
        """Generates plotly figure.

        Returns:
            go.Figure: figure
        """
        range_x = [1, len(self._source.index)]

        if self.similarity is not None:
            range_y = [0, 1]
        else:
            range_y = [0, len(self._source.index)]

        if self.percentage:
            range_x = [0, 1]
            range_y = [0, 1]

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
                range=range_x,
                linecolor="black",
                linewidth=1,
                tickangle=-90,
                tickfont=dict(family="IBM Plex Sans", size=11, color="black"),
            ),
            yaxis=dict(
                title="S",
                mirror=True,
                ticks="inside",
                range=range_y,
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
