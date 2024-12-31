import os
import time
from trackeval.eval import Evaluator
from trackeval import datasets
from trackeval.metrics import HOTA, Identity, CLEAR

#################
# DOES NOT WORK #
#################
# https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md
class Metrics:
    def __init__(self, gt_folder, tracker_folder, seq_name="ADL-Rundle-6"):
        self.gt_folder = gt_folder
        self.tracker_folder = tracker_folder
        self.seq_name = seq_name

        self.eval_config = {
            'DISPLAY_LESS_PROGRESS': False,
            'PRINT_CONFIG': True,
            'PRINT_RESULTS': True,
            'TIME_PROGRESS': True,
            'USE_PARALLEL': True,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True
        }

        self.dataset_config = {
            'GT_FOLDER': self.gt_folder,
            'TRACKER_FOLDER': self.tracker_folder,
            'OUTPUT_FOLDER': None,
            'TRACKERS_TO_EVAL': None,
            'CLASSES_TO_EVAL': ['pedestrian'],
            'BENCHMARK': '',
            'SPLIT_TO_EVAL': 'train',
            'INPUT_AS_ZIP': False,
            'PRINT_CONFIG': True,
            'DO_PREPROC': True,
            'TRACKER_SUB_FOLDER': 'data',
            'TRACKERS_FOLDER': self.tracker_folder,
            'SKIP_SPLIT_FOL': True,
            'GT_LOC_FORMAT': '{gt_folder}/gt.txt'
        }

    def evaluate(self):
        try:
            evaluator = Evaluator(self.eval_config)
            dataset_list = [datasets.MotChallenge2DBox(self.dataset_config)]
            metrics_list = [HOTA(), Identity(), CLEAR()]

            start_time = time.time()
            output = evaluator.evaluate(dataset_list, metrics_list)
            end_time = time.time()

            # Extract metrics
            metrics = {}
            seq_results = output['MotChallenge2DBox']['results'][self.seq_name]

            metrics['HOTA'] = seq_results['HOTA']['HOTA']
            metrics['IDF1'] = seq_results['Identity']['IDF1']
            metrics['ID_Switches'] = seq_results['Identity']['ID_switches']
            metrics['MOTA'] = seq_results['CLEAR']['MOTA']
            metrics['processing_time'] = end_time - start_time

            return metrics
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise