import yaml
from pathlib import Path
from typing import Dict, Union

import rich
import rich.tree
import rich.syntax

from omegaconf import DictConfig, OmegaConf


###################################
# functions
###################################

def load_config(config: Union[str, Dict, DictConfig]) -> Union[Dict, DictConfig]:
    if isinstance(config, str):
        config_path = Path(config).absolute()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found.")
        else:
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
    return config


def print_config(
    config: DictConfig,
    resolve: bool = False,
) -> None:

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)