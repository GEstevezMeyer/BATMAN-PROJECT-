import tomllib
import toml
import torch 
import pandas as pd 
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os 
import json 


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
            "labels": labels
        })

        return df

def create_loading_metadata(data_path:str) -> pd.DataFrame: 
    files = os.listdir(data_path)
    inputs = []
    labels = []
    ids = []

    for input in tqdm(files): 
        label = input.split("__")
        id = input.split(".")
        
        inputs.append(data_path+"/"+input)
        labels.append(label[0])
        ids.append(id[0])
    

    df = pd.DataFrame({
        "ids": ids,
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


def return_device():
    device = os.environ.get("device_torch")

    if device is None:
        return "cpu"

    return device

@named_action
def dump_history(history,res_path:str = "res/history.json"):
     with open(res_path,"w") as f:
          json.dump(history,f,indent=4)
     
def import_history(res_path:str = "res/history.json"): 
    with open(res_path,"rb") as f : 
        history = json.load(f)

    return history

@named_action
def save_model_weights(model, res_path="res"):
    torch.save(model.state_dict(), f"{res_path}/model.pth")


def load_model(architecture, model_weights_path="res/model.pth"):
    architecture.load_state_dict(torch.load(model_weights_path))

    return architecture
    

if __name__ == "__main__": 
    print(import_history())
