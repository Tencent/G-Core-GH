import os

import wandb
import yaml

_DEFAULT_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_map.yaml")
_WANDB_INTERNAL = ("_step", "_timestamp", "_runtime")


def _load_metric_map(map_path: str) -> dict:
    with open(map_path) as f:
        return yaml.safe_load(f)


def _translate_row(row: dict, metric_map: dict) -> dict:
    """Translate a wandb history row from gcore naming to verl naming.

    Mapped keys are renamed (one-to-many lists are fanned out to every verl
    key with the same value); keys that are absent from the map or mapped to
    null pass through with their original gcore name so no data is lost.
    """
    out = {}
    for key, value in row.items():
        if key in _WANDB_INTERNAL:
            continue
        target = metric_map.get(key, key)
        if target is None:
            target = key
        if isinstance(target, list):
            for verl_key in target:
                out[verl_key] = value
        else:
            out[target] = value
    return out


def copy_runs(
    src_paths: list[str],
    dst_project: str,
    entity: str = "plt2",
    map_path: str = _DEFAULT_MAP_PATH,
    name_prefix: str = "copy-",
) -> None:
    """Copy wandb runs, translating gcore metric names to verl naming.

    Parameters
    ----------
    src_paths : list[str]
        Source run paths (``entity/project/run_id``) to copy, one per experiment.
    dst_project : str
        Destination wandb project for the copied runs.
    entity : str
        Destination wandb entity.
    map_path : str
        Path to the gcore->verl ``metrics_map.yaml``.
    name_prefix : str
        Prefix prepended to each source run name for the copied run.
    """
    api = wandb.Api()
    metric_map = _load_metric_map(map_path)
    for src_path in src_paths:
        run = api.run(src_path)
        new_run = wandb.init(
            project=dst_project,
            entity=entity,
            config=run.config,
            name=f"{name_prefix}{run.name}",
        )
        for row in run.scan_history():
            step = row["_step"]
            metrics = _translate_row(row, metric_map)
            wandb.log(metrics, step=step)
        new_run.finish()


if __name__ == "__main__":
    src_paths = [
        "plt2/sapo_qwen3_4b/djdlnckm",
        "plt2/sapo_qwen3_4b/2ilcx5a3",
        "plt2/sapo_qwen3_4b/3j8e934c",
    ]
    copy_runs(src_paths, dst_project="gcore_verl_sapo_alignment")
