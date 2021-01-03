import os
import glob
import sys
import pickle
import shutil

import numpy as np
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import rs utils
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import python3 coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# All used metrics
METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
NUM_EPOCHS = 50
SAVE_EVERY = 1

# From COCOEval code
class COCOScorer(object):
    def __init__(self):
        print('init COCO-EVAL scorer')
            
    def score(self, GT, RES, IDs, result_file):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        #print('Tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # Set up scorers
        #print('Setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # Compute scores
        eval = {}
        self.final_results = []
        for scorer, method in scorers:
            print('Computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print("%s: %0.3f"%(method, score))
        
        print()
        # Collect scores by metrics
        for metric in METRICS:
            self.final_results.append(self.eval[metric])
        self.final_results = np.array(self.final_results)

        return self.eval
    
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

def read_prediction_file(prediction_file):
    """Helper function to read generated prediction files.
    """
    # Create dicts for ground truths and predictions
    gts_dict, pds_dict = {}, {}
    f = open(prediction_file, 'r')
    lines = f.read().split('\n')
    f.close()

    for i in range(0, len(lines) - 4, 4):
        id_line = lines[i+1]
        gt_line = lines[i+2]
        pd_line = lines[i+3]
            
        # Build individual ground truth dict
        curr_gt_dict = {}
        curr_gt_dict['image_id'] = id_line
        curr_gt_dict['cap_id'] = 0 # only 1 ground truth caption
        curr_gt_dict['caption'] = gt_line
        gts_dict[id_line] = [curr_gt_dict]
            
        # Build current individual prediction dict
        curr_pd_dict = {}
        curr_pd_dict['image_id'] = id_line
        curr_pd_dict['caption'] = pd_line
        pds_dict[id_line] = [curr_pd_dict]
    
    return gts_dict, pds_dict

def test():
    """Helper function to test on any dataset.
    """
    scorer = COCOScorer()
    max_scores = np.zeros((len(METRICS), ), dtype=np.float32)
    max_pred_file = None
    max_model_file = None
    for i in range(1, NUM_EPOCHS, SAVE_EVERY):
        prediction_file = os.path.join(ROOT_DIR, 'checkpoints', 'prediction', '{}_prediction.txt'.format(i))
        print('-'*30)
        print('Evaluating file:', prediction_file)
        gts_dict, pds_dict = read_prediction_file(prediction_file)
        ids = list(gts_dict.keys())
        scorer.score(gts_dict, pds_dict, ids, prediction_file)
        if np.sum(scorer.final_results) >= np.sum(max_scores):
            max_scores = scorer.final_results
            max_pred_file = '{}_prediction.txt'.format(i)
            max_model_file = 'v2l_epoch_{}.pth'.format(i)

    print('Maximum Score with file', max_model_file)
    
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    dir_path = os.path.join(ROOT_DIR, 'results_RS-RGBD', time_str)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f = open(os.path.join(dir_path, 'metric_result.txt'), 'w')
    for i in range(len(max_scores)):
        print('%s: %0.3f' % (METRICS[i], max_scores[i]))
        f.write('%s: %0.3f\n' % (METRICS[i], max_scores[i]))
    f.close()
    shutil.move(os.path.join(ROOT_DIR, 'checkpoints', 'saved', max_model_file), 
                os.path.join(dir_path, max_model_file))
    shutil.move(os.path.join(ROOT_DIR, 'checkpoints', 'prediction', max_pred_file), 
                os.path.join(dir_path, max_pred_file))
    

if __name__ == '__main__':
    test()