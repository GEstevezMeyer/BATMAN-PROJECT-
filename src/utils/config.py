import tomllib
import toml

import pandas as pd 
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os 


def named_action(func):

    def wrap(*args, **kwargs):

        print("-----------------")
        print(f"Executing: {func.__name__}")
        print("------------------")
        result = func(*args, **kwargs)
        print("------------------")
        print(f"Finishing: {func.__name__}")
        print("------------------")

        return result

    return wrap

@named_action
def import_config(toml_path:str = "config.toml") -> dict: 
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return data

@named_action
def dump_config(new_config, toml_path: str = "config.toml"):
    with open(toml_path, "w") as f:
        toml.dump(new_config, f)


@named_action
def create_metadata(data_path:str) -> pd.DataFrame: 
        files = os.listdir(data_path)
        inputs = []
        labels = []

        for input in tqdm(files): 
            inputs.append(data_path+"/"+input)
            label = input.split("__")
            labels.append(label[0])
    

        df = pd.DataFrame({
            "path": inputs,
            "label": labels
        })

        return df

@named_action
def compute_mean_std(data_path:str,toml_path:str = "config.toml"):
        
        metadata = create_metadata(data_path)
        config = import_config(toml_path)

        mean = 0.0
        sq_mean = 0.0
        n = 0

        for i in tqdm(range(len(metadata))):
            data = metadata.iloc[i]
            img = Image.open(data["path"]).convert("RGB")
            img = transforms.ToTensor()(img)  

            mean += img.mean(dim=(1, 2))
            sq_mean += (img ** 2).mean(dim=(1, 2))
            n += 1

        mean /= n
        sq_mean /= n

        std = (sq_mean - mean ** 2).sqrt()

        config["dataset"]["mean"] = mean.tolist()
        config["dataset"]["std"] = std.tolist()

        dump_config(config,toml_path)




     

if __name__ == "__main__": 
     compute_mean_std("DATA/SOCOFing/Real")
