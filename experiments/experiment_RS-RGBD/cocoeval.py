import os
import glob
import sys
import pickle

import numpy as np
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import python3 coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# All used metrics
METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

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
    # Get all generated predicted files
    prediction_files = sorted(glob.glob(os.path.join(ROOT_DIR, 'checkpoints', 'prediction', '*.txt')))
    
    scorer = COCOScorer()
    max_scores = np.zeros((len(METRICS), ), dtype=np.float32)
    max_file = None
    for prediction_file in prediction_files:
        gts_dict, pds_dict = read_prediction_file(prediction_file)
        ids = list(gts_dict.keys())
        scorer.score(gts_dict, pds_dict, ids, prediction_file)
        if np.sum(scorer.final_results) > np.sum(max_scores):
            max_scores = scorer.final_results
            max_file = prediction_file

    print('Maximum Score with file', max_file)
    fname = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.txt'
    f = open(fname, 'w')
    for i in range(len(max_scores)):
        print('%s: %0.3f' % (METRICS[i], max_scores[i]))
        f.write('%s: %0.3f\n' % (METRICS[i], max_scores[i]))
    f.close()
    

if __name__ == '__main__':
    test()