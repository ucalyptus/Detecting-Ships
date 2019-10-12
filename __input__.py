import json, sys, random
import numpy as np


def data():

    f = open(r"./input/shipsnet.json")
    dataset = json.load(f)
    f.close()

    input_data = np.array(dataset["data"]).astype("uint8")
    output_data = np.array(dataset["labels"]).astype(
        "uint8"
    )

    return (input_data, output_data)
