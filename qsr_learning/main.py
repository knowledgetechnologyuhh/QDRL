import argparse
import os
from functools import partial
from pathlib import Path

import git
import torch
from git.exc import RepositoryDirtyError
from munch import Munch
from ray import tune

from qsr_learning.relations.absolute.half_planes import left_of, right_of
from qsr_learning.train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
sha = repo.head.object.hexsha
ROOT = Path(repo.working_tree_dir)

config = Munch(
    model=Munch(
        embedding_dim=tune.grid_search([10, 20, 30, 40]),
    ),
    train=Munch(
        batch_size=64,
        num_epochs=100,
    ),
    data=Munch(
        negative_sample_mixture=Munch(head=1, relation=1, tail=1),
        train=Munch(
            relations=[left_of, right_of],
            num_images=tune.grid_search(
                [2 ** 13, 2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18]
            ),
            num_objects=3,
            num_pos_questions_per_image=2,
            num_neg_questions_per_image=2,
        ),
        validation=Munch(
            relations=[left_of, right_of],
            num_images=8096,
            num_objects=3,
            num_pos_questions_per_image=2,
            num_neg_questions_per_image=2,
        ),
        test=Munch(
            relations=[left_of, right_of],
            num_images=8096,
            num_objects=3,
            num_pos_questions_per_image=2,
            num_neg_questions_per_image=2,
        ),
    ),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debugging. Repo can be dirty."
    )
    args, _ = parser.parse_known_args()

    if not args.debug and repo.is_dirty():
        raise RepositoryDirtyError(repo, "Have you forgotten to commit the changes?")

    current_path = Path(repo.working_dir).as_posix()
    repo_name = repo.remotes.origin.url.split(".git")[0].split("/")[-1]
    if args.debug:
        repo_name = "[dirty]" + repo_name
    experiment_name = repo_name + "_" + sha[:8]

    analysis = tune.run(
        partial(train, device=device),
        metric="validation_accuracy",
        mode="max",
        name=experiment_name,
        stop={"training_iteration": 1} if args.smoke_test else None,
        config=config,
        resources_per_trial={"cpu": 9, "gpu": 1},
        num_samples=1 if args.smoke_test else 64,
        keep_checkpoints_num=1,
        checkpoint_score_attr="validation_accuracy",
    )
