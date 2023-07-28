import yaml
from yaml.loader import SafeLoader

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

config["variables"]["threshold"] = 100

with open("config.yml", "w") as f1:
    yaml.safe_dump(config, f1, default_flow_style=False)