import pickle
from tqdm import tqdm
from pprint import PrettyPrinter

with open("feature.pkl", "rb") as f:
    data: dict = pickle.load(f)


input_shape = data["train"]["input"].shape
label_shape = data["train"]["label"].shape

print(input_shape)
print(label_shape)