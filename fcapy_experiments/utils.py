import re
import os


def text_to_filename(text):
    first_replacement = re.sub(r"\[|\]|:|,|\)|\(", "", text.lower())

    return re.sub(r" ", "_", first_replacement)


def fig_to_file(fig, output_dir, output_filename):
    fig.write_image(os.path.join(
        output_dir, f"{output_filename}.pdf"), engine="kaleido")

    fig.write_image(os.path.join(
        output_dir, f"{output_filename}.png"), scale=3, engine="kaleido")
