import yaml

def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, sort_keys=False)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)