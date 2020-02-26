# Robot_Semantics
This is the official implementation decribed in our paper: 

"[Understanding Contexts Inside Joint Robot and Human Manipulation Tasks through Vision-Language Model with Ontology Constraints in a Video Streamline](?)"

## Requirements
- PyTorch (tested on 1.4)
- TorchVision with PIL
- numpy
- OpenCV (tested with 4.1.0)
- coco-caption, a [modified version](https://github.com/flauted/coco-caption/tree/python23) is used to support Python3
- Owlready2
- Graphviz

## Demos
We offer a pretrained model with our attention-seq2seq, download it [here]() and put it inside `robot_semantics/checkpoints/saved`.

Two files are provided for demo: (1) A [jupyter notebook]() to visualize the knowledge graph given outputs from the Vision-Language model. (2) `vis_att.py` to visualize all attention maps generated for a single video.

## Experiments
To repeat the experiments on our Robot Semantics Dataset:
1. Clone the repository.

2. Download the [Robot Semantics Dataset](?), check our wiki page for more details. Please extract the dataset and setup the directory path as `datasets/RS-RGBD`.

3. Select a branch to repreat the experiment (Please check our paper for detailed experiment settings). Under the folder `experiment_RS-RGBD/RS-RGBD`, run `generate_clips.py` to sample offline dataset videos into clips for training and evaluation.

4. To begin training, run `train.py`. Modify `v2c/config.py` accordingly to adjust the hyperparameters.

5. For evaluation, firstly run `evaluate.py` to generate predictions given all saved checkpoints. Run `cocoeval.py` to calculate scores for the predictions. `save_att.py` saves all attention maps for visualization demos.


## Additional Note
If you find this repository useful, please give me a star. Please leave me an issue if you find any potential bugs inside the code.

If you find this code useful, please consider citing:
```
Hello empty
```
