import pandas as pd
from fcapsy.typicality import typicality_avg
from fcapsy.similarity import jaccard, smc, rosch

from fcapsy_experiments._styles import css, css_typ


class ConceptTypicality:
    def __init__(
        self, concept, axis=0, human_judgements=None, typicality_functions=None
    ) -> None:
        if typicality_functions is None:
            typicality_functions = {
                "typ_avg": {
                    "func": typicality_avg,
                    "args": {"J": [jaccard], "SMC": [smc], "R": [rosch]},
                }
            }

        self.df = self._init(concept, axis, human_judgements, typicality_functions)

    @staticmethod
    def _init(concept, axis, human_judgements, typicality_functions):
        items = concept.extent

        if axis == 1:
            items = concept.intent

        columns = []

        for name, typicality in typicality_functions.items():
            for arg in typicality["args"].keys():
                columns.append(f"{name}({arg})")

        df = pd.DataFrame(index=items, columns=columns, dtype=float)

        for item in items:
            row = []
            for typicality in typicality_functions.values():
                function = typicality["func"]
                args = typicality["args"].values()
                row.extend([function(item, concept, *arg) for arg in args])

            df.loc[item] = row

        if human_judgements is not None:
            df["HJ"] = human_judgements

        return df

    def to_html(self):
        final_table = pd.DataFrame()

        for column in self.df.columns:
            round_and_sort = self.df.sort_values(column, ascending=False)

            final_table[f"{column} order"] = round_and_sort.index
            final_table[column] = round_and_sort.reset_index()[column]

        df = final_table.reset_index().drop("index", axis=1)

        df = df.style.format(precision=3)
        df.background_gradient(cmap="RdYlGn")
        df.set_table_styles(css + css_typ)
        df.hide_index()
        # df.set_caption(caption)

        return df.to_html()