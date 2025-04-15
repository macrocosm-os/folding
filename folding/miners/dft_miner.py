import os
import time
import glob
import copy
import json
import base64
import random
import hashlib
import requests
import traceback
import concurrent.futures
import asyncio

import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from folding.miners.folding_miner import FoldingMiner

import psi4


class DFTMiner(FoldingMiner):
    def __init__(self, config=None):
        super().__init__(config)

    def dft_forward(self, synapse: JobSubmissionSynapse) -> JobSubmissionSynapse:
        return synapse
