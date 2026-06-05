from argparse import ArgumentParser

import numpy as np
import wandb


def main(args):
    if args.names is not None:
        if args.runs is None:
            raise Exception("Renaming copied runs not supported when copying whole project.")
        assert len(args.names) == len(args.runs), "Number of new names must equal number of run IDs"

    args.dst_entity = args.dst_entity if args.dst_entity is not None else args.src_entity
    args.dst_project = args.dst_project if args.dst_project is not None else args.src_project
    same_project = args.src_entity == args.dst_entity and args.src_project == args.dst_project
    if same_project and args.names is None:
        name_append = "-copy"
    else:
        name_append = "-align-metrics"

    # Set your API key
    wandb.login()
    # Initialize the wandb API
    api = wandb.Api()

    # Get the runs from the source project
    runs = api.runs(f"{args.src_entity}/{args.src_project}")

    # Iterate through the runs and copy them to the destination project
    for run in runs:
        if args.runs is not None and run.id not in args.runs:
            continue
        # Get the run history and files
        history = run.history(samples=run.lastHistoryStep + 1)
        # system = run.history(samples=run.lastHistoryStep + 1, stream="system")
        # history = history.join(system, rsuffix="_system")
        files = run.files()

        name = run.name if args.names is None else args.names[args.runs.index(run.id)]

        # Create a new run in the destination project

        # Log the history to the new run
        new_run = wandb.init(
            project=args.dst_project,
            entity=args.dst_entity,
            config=run.config,
            name=name + name_append,
            resume="allow"
        )

        # make step 0 empty
        new_run.log({})
        for index, row in history.iterrows():
            # print(f"{index=} {row=}")
            item = {
                k: v
                for k, v in row.to_dict().items() if v is None or not (v == "NaN" or np.isnan(v))
            }
            new_item = {}
            useless_metrics = {"train/epoch"}
            replace_map = {
                "train/loss": "lm loss",
                "train/grad_norm": "grad-norm",
                "train/learning_rate": "learning-rate",
                "train/global_step": "step",
            }
            for k, v in item.items():
                if k.startswith("system") or k in useless_metrics:
                    continue
                if k == "_step":
                    # print(f"{k=} {v=} {index=}")
                    new_item[k] = v + 1
                    continue
                if k in replace_map:
                    new_item[replace_map[k]] = v
                else:
                    new_item[k] = v
            new_run.log(new_item)

        # Upload the files to the new run
        for file in files:
            file.download(replace=True)
            new_run.save(file.name, policy="now")

        # Finish the new run
        new_run.finish()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Copies one or all of the runs in a wandb project to another."
    )
    parser.add_argument(
        "-se", "--src-entity", type=str, default="indezera", help="Source wandb entity name."
    )
    parser.add_argument("-sp", "--src-project", type=str, help="Name of the wandb projecet.")
    parser.add_argument(
        "-de", "--dst-entity", type=str, default=None, help="Destination wandb entity name."
    )
    parser.add_argument(
        "-dp", "--dst-project", type=str, default=None, help="Name of destination wandb project."
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="*",
        type=str,
        default=None,
        help="List of run IDs to copy. If None will copy all in project."
    )
    parser.add_argument(
        "-n",
        "--names",
        nargs="*",
        type=str,
        default=None,
        help="List of new names for copied runs (optional)."
    )

    main(parser.parse_args())
