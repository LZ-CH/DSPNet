import os
from typing import Dict, List, Optional, Sequence
import json
import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable
import numpy as np
from embodiedqa.registry import METRICS
from embodiedqa.structures import EulerDepthInstance3DBoxes

@METRICS.register_module()
class ScanQAMetric(BaseMetric):
    def __init__(self,
                 topk: List[float] = [1, 10],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only=False,
                 extra_pred_scores_suffix = None,
                 question_type_analysis=True,
                 result_dir='') -> None:
        super(ScanQAMetric, self).__init__(prefix=prefix,
                                            collect_device=collect_device)
        self.topk = topk
        self.prefix = prefix
        self.format_only = format_only
        self.extra_pred_scores_suffix =  extra_pred_scores_suffix
        self.result_dir = result_dir
        self.question_type_analysis = question_type_analysis
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred = dict(pred_scores=data_sample['pred_scores'].to('cpu'))
            if self.extra_pred_scores_suffix is not None:
                for suffix in self.extra_pred_scores_suffix:
                    if f'pred_scores{suffix}' in data_sample:
                        cpu_pred[f'pred_scores{suffix}'] = data_sample[f'pred_scores{suffix}'].to('cpu')
            self.results.append((eval_ann_info, cpu_pred))

    def ground_eval(self, gt_annos, pred_annos, logger=None,type_suffix=''):

        assert len(pred_annos) == len(gt_annos)
        pred = {}
        gt = {}

        metric_types = ['EM@'+str(k) for k in self.topk]
        if self.question_type_analysis:
            metric_types +=  ['what','where','how','is','which','others']
        metric_types = [t + type_suffix for t in metric_types]
        for metric_type in metric_types:
            pred.update({metric_type: 0})
            gt.update({metric_type: 0})

        for sample_id in range(len(pred_annos)):
            pred_anno = pred_annos[sample_id]
            gt_anno = gt_annos[sample_id]
            pred_scores = pred_anno['pred_scores']  # (num_cls, )
            gt_answer_labels = gt_anno['gt_answer_labels']
            top_index = pred_scores.argsort(dim=-1, descending=True)[:max(self.topk)]
            if self.question_type_analysis:
                question_type = gt_anno['question_type']
                gt[question_type+type_suffix] += 1
                pred[question_type+type_suffix] += int((gt_answer_labels[top_index[:1]]).any())
            for k in self.topk:
                found = int((gt_answer_labels[top_index[:k]]).any())
                gt['EM@' + str(k) + type_suffix] += 1
                pred['EM@' + str(k) + type_suffix] += found

        header = ['Type']
        header.extend(metric_types)
        ret_dict = {}

        table_columns = [['results']]
        for metric_type in metric_types:
            value = pred[metric_type] / max(gt[metric_type], 1)
            ret_dict[metric_type] = value
            table_columns.append([f'{value:.4f}'])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        return ret_dict

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results after all batches have
        been processed.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()  # noqa

        annotations, preds = zip(*results)
        ret_dict = {}
        
        # preds is a list of dict
        results = []
        answer_candidates = self.dataset_meta.get('answer_candidates')
        for i, pred in enumerate(preds):
            gt_answer_id = np.where(annotations[i]['gt_answer_labels']==1)[0]
            # convert the Euler boxes to the numpy array to save
            pred_scores = pred['pred_scores']
            # eval top-10 predictions during the test phase by default
            top10_index = pred_scores.argsort(dim=-1, descending=True)[:10]
            result = dict(question=annotations[i]['question'],
                          question_id=annotations[i]['question_id'],
                          answer_top10=[answer_candidates[k] for k in top10_index],
                          scene_id=annotations[i]['scan_id'].split('/')[-1],
                          )
            if not self.format_only:
                result['gt_answer']=[answer_candidates[k] for k in gt_answer_id]                
            results.append(result)
        with open(os.path.join(self.result_dir, 'test_results.json'), 'w') as f:
            json.dump(results,f, indent=4)
        if self.format_only:
            return ret_dict

        ret_dict = self.ground_eval(annotations, preds, logger=logger)
        
        if self.extra_pred_scores_suffix is not None:
            for suffix in self.extra_pred_scores_suffix:
                if f'pred_scores{suffix}' in preds[0]:
                    ret_dict.update(self.ground_eval(annotations, preds, logger=logger ,type_suffix=suffix))
        return ret_dict 