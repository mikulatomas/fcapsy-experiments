import pandas as pd
from fcapsy.typicality import typicality_avg
from fcapsy.similarity import jaccard, smc, rosch


def concept_typicality(concept, axis=0, ground_truth=None, typicality_functions=None):
    if typicality_functions is None:
        typicality_functions = {
            "typ_avg": {
                "func": typicality_avg,
                "args": {"J": [jaccard], "SMC": [smc], "R": [rosch]},
            }
        }

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
    
    return df

