from mmengine.fileio import load
import argparse
anno_path = '/data1/luojingzhou/projects/EmbodyAI/EmbodiedQA/data/sqa_task/balanced/v1_balanced_sqa_annotations_test_scannetv2.json'
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pred_path', type=str, default=None, required=True)
    args.add_argument('--anno_path', type=str, default=anno_path)
    args = args.parse_args()
    pred_json = load(args.pred_path)
    anno_json = load(args.anno_path)
    if isinstance(anno_json, dict): #SQA3D
        anno_json = anno_json['annotations']
        data_type = "sqa3d"
    elif isinstance(anno_json, list): #ScanQA
        data_type = "scanqa"
    top1_correct_count = 0
    top10_correct_count = 0
    assert len(pred_json)==len(anno_json)
    for p,t in zip(pred_json, anno_json):
        assert p['question_id'] == t['question_id']
        if data_type=="sqa3d":
            anno_answers = [a['answer'] for a in t['answers']]
        elif data_type=="scanqa":
            anno_answers = t['answers']
        if p['answer_top10'][0] in anno_answers:
            top1_correct_count += 1
        if any([p['answer_top10'][i] in anno_answers for i in range(10)]):
            top10_correct_count += 1
    print("EM@1:", top1_correct_count/len(pred_json))
    print("EM@10:", top10_correct_count/len(pred_json))