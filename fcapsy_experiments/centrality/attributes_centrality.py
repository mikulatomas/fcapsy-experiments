import pandas as pd

from fcapsy.centrality import attribute_centrality
from fcapsy_experiments._styles import css, css_corr

class AttributesCentrality:
    def __init__(self, concept, extra_columns=None, intent_indicator=False) -> None:
        self.df = self._init(concept, extra_columns, intent_indicator)
        self._concept = concept

        if extra_columns:
            extra_columns = list(extra_columns.keys())
        
        self.extra_columns = extra_columns

    @staticmethod
    def _init(concept, extra_columns, intent_indicator):
        context = concept.lattice._context

        df = pd.DataFrame(
            [attribute_centrality(attr, concept) for attr in context.properties],
            index=context.properties,
            columns=["Centrality"],
        )

        if extra_columns:
            for name, values in extra_columns.items():
                df[name] = values
        
        if intent_indicator:
            df["is intent"] = [int(attr in concept.intent) for attr in context.properties]

        return df
    
    def _filter_sort_df(self, intent=False, quantile=0.75):
        filtered_df = self.df.loc[self.df["Centrality"] > 0]

        if not intent:
            filtered_df = filtered_df.drop(list(self._concept.intent), axis=0)

        quantile_value = filtered_df.quantile(quantile)["Centrality"]
        filtered_df = filtered_df.loc[filtered_df["Centrality"] >= quantile_value]
        filtered_df = filtered_df.sort_values("Centrality", ascending=False)

        return filtered_df

    def to_html(self, intent=False, quantile=0.75):
        final_table = self._filter_sort_df(intent=intent, quantile=quantile)

        final_table = final_table.style.format(precision=3)
        final_table.background_gradient(cmap="RdYlGn")
        final_table.set_table_styles(css + css_corr)

        return final_table.to_html()
