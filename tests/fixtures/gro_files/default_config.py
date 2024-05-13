import argparse
import bittensor as bt
from folding.utils.config import add_args, add_miner_args


def alter_config(parser):
    for index, action in enumerate(parser._actions):
        if action.dest in ["mock"]:
            parser._actions[index].default = "True"

        if action.dest in ["protein.pdb_id"]:
            parser._actions[index].default = "test_pdb_id"

    return parser


def get_test_config():
    PARSER = argparse.ArgumentParser()
    add_args(None, PARSER)
    add_miner_args(None, PARSER)
    PARSER = alter_config(PARSER)

    return bt.config(PARSER)
