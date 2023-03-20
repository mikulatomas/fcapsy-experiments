import typing

import pandas as pd

from itertools import combinations

from .top_r_similarity import TopRSimilarity


class TopBottomRSimilarity(TopRSimilarity):
    @staticmethod
    def _init(inst, to_columns):
        r_range = range(1, len(inst._source.index) + 1)

        results = []

        columns_tuples, labels = inst._get_column_orders(
            inst._source,
            [(x, y) for x in inst._source.columns for y in to_columns if x != y],
        )

        for (column1_order, column2_order), label in zip(columns_tuples, labels):
            column1_order_reversed = column1_order[::-1]
            column2_order_reversed = column2_order[::-1]

            for r in r_range:
                top_r = inst._top_r_similarity(
                    inst._context, inst._similarity, column1_order, column2_order, r
                )
                bottom_r = inst._top_r_similarity(
                    inst._context,
                    inst._similarity,
                    column1_order_reversed,
                    column2_order_reversed,
                    r,
                )
                results.append(
                    [
                        r,
                        top_r * bottom_r,
                        label,
                    ]
                )

        return pd.DataFrame(results, columns=["r", "top_bottom_r_similarity", "label", "a", "b"])
