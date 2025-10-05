from copy import deepcopy
from pathlib import Path
import pprint
import time

from worldscore.benchmark.utils.utils import aspect_info as ASPECT_INFO
from worldscore.benchmark.helpers.evaluator import renormalize_score
from worldscore.benchmark.metrics import CLIPScoreMetric, CLIPImageQualityAssessmentPlusMetric, IQACLIPAestheticScoreMetric, GramMatrixMetric,\
    CameraErrorMetric, OpticalFlowAverageEndPointErrorMetric,\
    ObjectDetectionMetric, ReprojectionErrorMetric, OpticalFlowMetric


class MlCupMetricScorer:
    SKIP_ASPECTS = ['motion_accuracy', 'motion_smoothness', 'motion_magnitude']
    SKIP_METRICS = [
        'camera_error', # TODO: camera intrinsics
        'reprojection_error', # TODO: CUDA error
        'object_detection', # interface: no prompt
        'clip_score' # interface: no prompt
    ]

    METRIC_CLASSES = {
        "clip_iqa+": CLIPImageQualityAssessmentPlusMetric,
        "clip_aesthetic": IQACLIPAestheticScoreMetric,
        "gram_matrix": GramMatrixMetric,
        "optical_flow_aepe": OpticalFlowAverageEndPointErrorMetric,
        "camera_error": CameraErrorMetric,
        "object_detection": ObjectDetectionMetric,
        "reprojection_error": ReprojectionErrorMetric,
        "optical_flow": OpticalFlowMetric,
        "clip_score": CLIPScoreMetric,
    }

    def __init__(self):
        self._metrics = deepcopy(ASPECT_INFO)

    def __call__(self, image_paths: list[str]):
        metrics_result = {}
        aspects_scores = []
        t_init = time.time()
        for aspect, aspect_data in ASPECT_INFO.items():
            if aspect in self.SKIP_ASPECTS:
                continue
            metrics_result[aspect] = {}
            current_aspect_scores = []
            for metric_name, metric_data in aspect_data['metrics'].items():
                if metric_name in self.SKIP_METRICS:
                    continue
                t_start = time.time()
                scorer_cls = self.METRIC_CLASSES[metric_name]
                scorer = scorer_cls()
                args = [image_paths]
                if metric_name == "gram_matrix":
                    args = [image_paths[0], image_paths[1:]]
                score_value = scorer._compute_aspect_scores(*args)
                current_aspect_scores.append(score_value)
                metrics_result[aspect][metric_name] = {'score' : score_value, 'time' : time.time() - t_start}

            if len(current_aspect_scores) > 0:
                aspects_scores.append(sum(current_aspect_scores) / (len(current_aspect_scores)))

        metrics_result = renormalize_score(metrics_result)
        metrics_result['total_score'] = sum(aspects_scores) / (len(aspects_scores) + 1e-6)
        metrics_result['total_time'] = time.time() - t_init
        return metrics_result


if __name__ == "__main__":
    import sys
    assert len(sys.argv) > 1, sys.argv
    for folder_path in sys.argv[1:]:
        root = Path(folder_path).absolute()
        images = list([str(x.absolute()) for x in root.glob("*.png")])
        mlcup_scorer = MlCupMetricScorer()
        result = mlcup_scorer(images)
        print("=========")
        print(folder_path)
        pprint.pprint(result, compact=True)