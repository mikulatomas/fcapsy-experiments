import pandas as pd

from fcapsy.centrality import centrality
from fcapsy_experiments._styles import css, css_corr

class Centrality:
    def __init__(self, concept, axis=0, extra_columns=None, core_indicator=False) -> None:
        self._concept = concept
        self.axis = axis
        self.df = self._init(extra_columns, core_indicator)
        
        if extra_columns:
            extra_columns = list(extra_columns.keys())
        
        self.extra_columns = extra_columns

    def _init(self, extra_columns, core_indicator):
        context = self._concept.lattice._context

        if self.axis == 0:
            items_domain = context.objects
            concept_core = self._concept.extent
        elif self.axis == 1:
            items_domain = context.properties
            concept_core = self._concept.intent
        else:
            raise ValueError("Invalid axis index")
        
        df = pd.DataFrame(
            [centrality(item, self._concept) for item in items_domain],
            index=items_domain,
            columns=["Centrality"],
        )

        if extra_columns:
            for name, values in extra_columns.items():
                df[name] = values
        
        if core_indicator:
            df["is core"] = [int(item in concept_core) for item in items_domain]

        return df
    
    def _filter_sort_df(self, include_core_flag=False, quantile=0.75):
        filtered_df = self.df.loc[self.df["Centrality"] > 0]

        if self.axis == 0:
            concept_core = self._concept.extent
        else:
            concept_core = self._concept.intent

        if not include_core_flag:
            filtered_df = filtered_df.drop(list(concept_core.members()), axis=0)

        quantile_value = filtered_df.quantile(quantile)["Centrality"]
        filtered_df = filtered_df.loc[filtered_df["Centrality"] >= quantile_value]
        filtered_df = filtered_df.sort_values("Centrality", ascending=False, kind="mergesort")

        return filtered_df

    def to_html(self, include_core_flag=False, quantile=0.75):
        final_table = self._filter_sort_df(include_core_flag=include_core_flag, quantile=quantile)

        final_table = final_table.style.format(precision=3)
        final_table.background_gradient(cmap="RdYlGn")
        final_table.set_table_styles(css + css_corr)

        return final_table.to_html()
