import typing

import pandas as pd
import plotly.express as px

from .top_r_intersection import TopRIntersection


class TopBottomRIntersection(TopRIntersection):
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
            column1_order_reversed = column1_order[::-1]
            column2_order_reversed = column2_order[::-1]

            for r in r_range:
                top_r = inst._top_r_intersection(column1_order, column2_order, r)
                bottom_r = inst._top_r_intersection(
                    column1_order_reversed,
                    column2_order_reversed,
                    r
                )

                intersection_size = top_r + bottom_r

                if percentage:
                    intersection_size = (top_r + bottom_r) / (2 * r)

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
            results, columns=["r", "top_bottom_r_intersection", "label", "a", "b"]
        )