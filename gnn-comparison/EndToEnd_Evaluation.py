#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import os

import torch

from config.base import Grid, Config
from evaluation.model_selection.HoldOutSelector import HoldOutSelector
from evaluation.risk_assessment.K_Fold_Assessment import KFoldAssessment
from experiments.EndToEndExperiment import EndToEndExperiment

from datasets.manager import (Proteins, 
NCI1, 
Enzymes, 
IMDBBinary, 
RedditBinary,
IMDBMulti,Collab, DD) # added DD here

def main(config_file, dataset_name,
         outer_k, outer_processes, inner_k, inner_processes, result_folder, debug=False):

    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine
    torch.set_num_threads(1)

    experiment_class = EndToEndExperiment

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    

    dataset_class_map = {
    "PROTEINS": Proteins,
    "ENZYMES": Enzymes,
    "NCI1": NCI1,
    "REDDIT-BINARY": RedditBinary,
    "IMDB-BINARY": IMDBBinary,
    "IMDB-MULTI": IMDBMulti,
    "COLLAB": Collab,
    "DD": DD # this line was added
}
    dataset_class = dataset_class_map.get(dataset_name.upper())

    gdm = dataset_class()
    print(dataset_name)
    metadata = getattr(gdm, "metadata", {})
    print(metadata)
    # you can add anything you like here manually, for example:
    metadata["rewiring_method"] = metadata.get("rewiring_strategy", "none")
    metadata["self_loops_added"] = metadata.get("add_self_loops", False)
    use_one = metadata.get("use__one", False)
    use_node_degree = metadata.get("use_node_degree", False)
    

    exp_path = os.path.join(result_folder, f'{model_configuration.exp_name}_{metadata["rewiring_method"]}_self_loops_{metadata["self_loops_added"]}'  + (f'_use_one' if use_one else '')
    + (f'_use_node_degree' if use_node_degree else ''))

    model_selector = HoldOutSelector(max_processes=inner_processes)
    risk_assesser = KFoldAssessment(outer_k, model_selector, exp_path, model_configurations,
                                    outer_processes=outer_processes)

    risk_assesser.risk_assessment(experiment_class, debug=debug)