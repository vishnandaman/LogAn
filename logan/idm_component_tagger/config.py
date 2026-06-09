import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tags=[]
    for entry in config.get("tags", []):
        tags.append({
            "name": entry["name"],
            "keywords": entry.get("keywords",[]),
            "patterns": entry.get("patterns",[]),
        })
    
    return {
        "tags": tags,
        "default_tag": config.get("default_tag", "Other"),
    }
