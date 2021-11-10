# import pandas as pd

# from fcapsy import Context, Concept
# from fcapsy.psychology.basic_level import basic_level_avg
# from fcapsy.psychology.cohesion import cohesion_avg
# from fcapsy.similarity import smc, jaccard
# from fcapsy.algorithms.lindig import upper_neighbors, lower_neighbors

# from fcapsy_experiments.utils import text_to_filename, fig_to_file
# from fcapsy_experiments.general_plots import plot_table


# def exp_concept_basic_level(concepts,
#                             context,
#                             basic_level_metrics={
#                                 basic_level_avg: {cohesion_avg: [smc, jaccard]}
#                             }):
#     columns = []

#     for basic_level, arguments in basic_level_metrics.items():
#         for cohesion, similarity_functions in arguments.items():
#             columns.extend([
#                 f"{basic_level.short_name}({cohesion.short_name}({similarity_function.short_name}))" for similarity_function in similarity_functions])

#     df = pd.DataFrame(
#         columns=columns,
#         index=[concept.name for concept in concepts],
#         dtype=float)

#     for concept in concepts:
#         row = []

#         upper_neighbor = tuple(upper_neighbors(context, concept))
#         lower_neighbor = tuple(lower_neighbors(context, concept))

#         for basic_level, arguments in basic_level_metrics.items():
#             for cohesion, similarity_functions in arguments.items():
#                 row.extend([basic_level(concept, context, upper_neighbor, lower_neighbor, cohesion, similarity_function)
#                             for similarity_function in similarity_functions])

#         df.loc[concept.name] = row

#     return df


# def plot_basic_level(df, title, index_title="", decimals=3, width=None, output_dir=None):
#     if width is None:
#         width = len(df.columns) * 220

#     if output_dir:
#         output_filename = f"{plot_basic_level.__name__}_{text_to_filename(title)}"

#         if index_title != "":
#             output_filename = f"{output_filename}_{index_title}"
#     else:
#         output_filename = None

#     return plot_table(df, title, index_title=index_title, decimals=decimals, width=width, output_dir=output_dir, output_filename=output_filename)
