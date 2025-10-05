from pathlib import Path
import pprint
import time

from worldscore.benchmark.utils.utils import aspect_info as ASPECT_INFO
from worldscore.benchmark.helpers.evaluator import renormalize_score
from worldscore.benchmark.metrics import CLIPScoreMetric, CLIPImageQualityAssessmentPlusMetric, IQACLIPAestheticScoreMetric, GramMatrixMetric,\
    CameraErrorMetric, OpticalFlowAverageEndPointErrorMetric,\
    ObjectDetectionMetric, ReprojectionErrorMetric, OpticalFlowMetric

RAISE_FAILS = False


class MlCupMetricScorer:
    SKIP_ASPECTS = ['motion_accuracy', 'motion_smoothness', 'motion_magnitude']
    SKIP_METRICS = [
        'camera_error', # TODO: camera intrinsics
        'object_detection', # interface: no prompt
        'clip_score' # interface: no prompt
        #'reprojection_error', works but useless
    ]

    def __init__(self):
        self._scorers = {
            "clip_iqa+": CLIPImageQualityAssessmentPlusMetric(),
            "clip_aesthetic": IQACLIPAestheticScoreMetric(),
            "gram_matrix": GramMatrixMetric(),
            "optical_flow_aepe": OpticalFlowAverageEndPointErrorMetric(),
            "optical_flow": OpticalFlowMetric(),
            #"camera_error": CameraErrorMetric(),
            #"object_detection": ObjectDetectionMetric(),
            #"reprojection_error": ReprojectionErrorMetric(),
            #"clip_score": CLIPScoreMetric(),
        }

    def __call__(self, image_paths: list[str]):
        metrics_result = {}
        t_init = time.time()
        for aspect, aspect_data in ASPECT_INFO.items():
            if aspect in self.SKIP_ASPECTS:
                continue
            metrics_result[aspect] = {}
            for metric_name, metric_data in aspect_data['metrics'].items():
                if metric_name in self.SKIP_METRICS:
                    continue
                args = [image_paths]
                if metric_name == "gram_matrix":
                    args = [image_paths[0], image_paths[1:]]

                t_start = time.time()
                try:
                    score_value = self._scorers[metric_name]._compute_scores(*args)
                    metrics_result[aspect][metric_name] = {'score' : score_value, 'time' : time.time() - t_start}
                except Exception as exc:
                    if RAISE_FAILS:
                        raise
        metrics_result = renormalize_score(metrics_result)
        aspect_normalized_scores = []
        for aspect, aspect_scores in metrics_result.items():
            if len(aspect_scores) > 0:
                print(aspect_scores)
                aspect_normalized_scores.append(
                    sum(m['score_normalized'] for m in aspect_scores.values()) / len(aspect_scores)
                )
        metrics_result['avg_normalized_score'] = sum(aspect_normalized_scores) / len(aspect_normalized_scores)
        metrics_result['total_time'] = time.time() - t_init
        return metrics_result


if __name__ == "__main__":
    import sys
    assert len(sys.argv) > 1, sys.argv
    t0 = time.time()
    mlcup_scorer = MlCupMetricScorer()
    for folder_path in sys.argv[1:]:
        root = Path(folder_path).absolute()
        images = list([str(x.absolute()) for x in root.glob("*.png")])
        result = mlcup_scorer(images)
        print("=========")
        print(folder_path)
        pprint.pprint(result, compact=True)
    print(f"Total time: {time.time() - t0}")