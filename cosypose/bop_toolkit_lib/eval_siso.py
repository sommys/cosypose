# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019/2020."""

import os
import time
import argparse
import subprocess
import numpy as np

from . import config
from . import inout
from . import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {  # pylint: disable=invalid-name
    # Errors to calculate.
    "errors": [
        {
            "n_top": 1,
            "type": "vsd",
            "vsd_deltas": {
                "hb": 15,
                "icbin": 15,
                "icmi": 15,
                "itodd": 5,
                "lm": 15,
                "lmo": 15,
                "ruapc": 15,
                "tless": 15,
                "tudl": 15,
                "tyol": 15,
                "ycbv": 15,
            },
            "vsd_taus": [20],
            "vsd_normalized_by_diameter": False,
            "correct_th": [[0.3]],
        },
        {
            "n_top": 1,
            "type": "adi",
            "correct_th": [[0.1]],
        },
    ],
    # Minimum visible surface fraction of a valid GT pose.
    # -1 == k most visible GT poses will be considered, where k is given by
    # the "inst_count" item loaded from "targets_filename".
    "visib_gt_min": 0.1,
    # See misc.get_symmetry_transformations().
    "max_sym_disc_step": 0.01,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "python",  # Options: 'cpp', 'python'.
    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # http://ptak.felk.cvut.cz/6DB/public/bop_sample_results/bop_challenge_2019/
    "result_filenames": [
        "/relative/path/to/csv/with/results",
    ],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    # Pix2pose file
    # https://github.com/kirumang/Pix2Pose
    "targets_filename": "all_target_tless.json",
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--renderer_type", default=p["renderer_type"])
parser.add_argument(
    "--result_filenames",
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--eval_path", default=p["eval_path"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
args = parser.parse_args()

p["renderer_type"] = str(args.renderer_type)
p["result_filenames"] = args.result_filenames.split(",")
p["results_path"] = str(args.results_path)
p["eval_path"] = str(args.eval_path)
p["targets_filename"] = str(args.targets_filename)

# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in p["result_filenames"]:

    misc.log("===========")
    misc.log(f"EVALUATING: {result_filename}")
    misc.log("===========")

    time_start = time.time()

    # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD).
    average_recalls = {}

    # Name of the result and the dataset.
    result_name = os.path.splitext(os.path.basename(result_filename))[0]
    dataset = result_name.split("_")[1].split("-")[0]  # pylint: disable=invalid-name

    # Calculate the average estimation time per image.
    ests = inout.load_bop_results(
        os.path.join(p["results_path"], result_filename), version="bop19"
    )
    times = {}
    times_available = True  # pylint: disable=invalid-name
    for est in ests:
        result_key = f"{est['scene_id']:06d}_{est['im_id']:06d}"
        if est["time"] < 0:
            # All estimation times must be provided.
            times_available = False  # pylint: disable=invalid-name
            break
        elif result_key in times:
            if abs(times[result_key] - est["time"]) > 0.001:
                raise ValueError(
                    f"The running time for scene {est['scene_id']} and image {est['im_id']} "
                    "is not the same for all estimates."
                )
        else:
            times[result_key] = est["time"]

    if times_available:
        average_time_per_image = np.mean(list(times.values()))
    else:
        average_time_per_image = -1.0  # pylint: disable=invalid-name

    # Evaluate the pose estimates.
    for error in p["errors"]:

        # Calculate error of the pose estimates.
        calc_errors_cmd = [
            "python",
            os.path.join("scripts", "eval_calc_errors.py"),
            f"--n_top={error['n_top']}",
            f"--error_type={error['type']}",
            f"--result_filenames={result_filename}",
            f"--renderer_type={p['renderer_type']}",
            f"--results_path={p['results_path']}",
            f"--eval_path={p['eval_path']}",
            f"--targets_filename={p['targets_filename']}",
            f"--max_sym_disc_step={p['max_sym_disc_step']}",
            "--skip_missing=1",
        ]
        if error["type"] == "vsd":
            vsd_deltas_str = ",".join(  # pylint: disable=invalid-name
                [f"{k}:{v}" for k, v in error["vsd_deltas"].items()]
            )
            calc_errors_cmd += [
                f"--vsd_deltas={vsd_deltas_str}",
                f"--vsd_taus={','.join(map(str, error['vsd_taus']))}",
                f"--vsd_normalized_by_diameter={error['vsd_normalized_by_diameter']}",
            ]

        misc.log(f"Running: {' '.join(calc_errors_cmd)}")
        if subprocess.call(calc_errors_cmd) != 0:
            raise RuntimeError("Calculation of pose errors failed.")

        # Paths (rel. to p['eval_path']) to folders with calculated pose errors.
        # For VSD, there is one path for each setting of tau. For the other pose
        # error functions, there is only one path.
        error_dir_paths = {}
        if error["type"] == "vsd":
            for vsd_tau in error["vsd_taus"]:
                error_sign = misc.get_error_signature(
                    error["type"],
                    error["n_top"],
                    vsd_delta=error["vsd_deltas"][dataset],
                    vsd_tau=vsd_tau,
                )
                error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
        else:
            error_sign = misc.get_error_signature(error["type"], error["n_top"])
            error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

        # Recall scores for all settings of the threshold of correctness (and also
        # of the misalignment tolerance tau in the case of VSD).
        recalls = []

        # Calculate performance scores.
        for error_sign, error_dir_path in error_dir_paths.items():
            for correct_th in error["correct_th"]:

                calc_scores_cmd = [
                    "python",
                    os.path.join("scripts", "eval_calc_scores.py"),
                    f"--error_dir_paths={error_dir_path}",
                    f"--eval_path={p['eval_path']}",
                    f"--targets_filename={p['targets_filename']}",
                    f"--visib_gt_min={p['visib_gt_min']}",
                ]

                calc_scores_cmd += [
                    f"--correct_th_{error['type']}={','.join(map(str, correct_th))}",
                ]

                misc.log(f"Running: {' '.join(calc_scores_cmd)}")
                if subprocess.call(calc_scores_cmd) != 0:
                    raise RuntimeError("Calculation of scores failed.")

                # Path to file with calculated scores.
                score_sign = misc.get_score_signature(  # pylint: disable=invalid-name
                    correct_th, p["visib_gt_min"]
                )

                scores_filename = (  # pylint: disable=invalid-name
                    f"scores_{score_sign}.json"
                )
                scores_path = os.path.join(
                    p["eval_path"], result_name, error_sign, scores_filename
                )

                # Load the scores.
                misc.log(f"Loading calculated scores from: {scores_path}")
                scores = inout.load_json(scores_path)
                recalls.append(scores["recall"])

        average_recalls[error["type"]] = np.mean(recalls)

        misc.log(f"Recall scores: {' '.join(map(str, recalls))}")
        misc.log(f"Average recall: {average_recalls[error['type']]}")

    time_total = time.time() - time_start
    misc.log(f"Evaluation of {result_filename} took {time_total}s.")

misc.log("Done.")
