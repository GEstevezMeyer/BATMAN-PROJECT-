import psycopg
import tomllib




def import_config(config_path:str = "config.toml") -> dict: 
    with open(config_path,"rb") as r: 
        config = tomllib.load(r)
    
    return config 

def connect_database(config_path: str = "config.toml"): 
    config_database = import_config(config_path)["database"]
    conn =  psycopg.connect(**config_database)

    return conn 

