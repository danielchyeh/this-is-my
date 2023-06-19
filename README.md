# *This-Is-My* Dataset
As one of contributions in Meta-Personalizing Vision-Language Models To Find Named Instances in Video (CVPR 2023)

<a href="https://danielchyeh.github.io/metaper/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue" height=20.5></a>
<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Yeh_Meta-Personalizing_Vision-Language_Models_To_Find_Named_Instances_in_Video_CVPR_2023_paper.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=Link&color=green" height=20.5></a>
<a href="https://www.youtube.com/watch?v=DnOOThEGZmU&t=1s"><img src="https://img.shields.io/static/v1?label=Project&message=Video&color=red" height=20.5></a>

> [Chun-Hsiao Yeh](https://danielchyeh.github.io/),
> [Bryan Russell](https://bryanrussell.org/),
> [Josef Sivic](http://people.ciirc.cvut.cz/~sivic/),
> [Fabian Caba Heilbron](https://fabiancaba.com/),
> [Simon Jenni](https://sjenni.github.io/)<br>
> UC Berkeley, CIIRC CTU, Adobe Research<br>
> In CVPR 2023

<img src="https://github.com/danielchyeh/metaper/blob/main/assets/metaper-thisismy.png" alt="image" width="100%">

Examples from This-Is-My: Meta-Personalization D (top) vs Test-time personalization P (bottom-left) vs Query-time Q (bottom-right) datasets. In the Query-time dataset (bottom-right), we design a challenging video instance retrieval task. 
For example, the named instance (i.e., Alex's piano) is in the background and is barely visible, and for "Zak's dog Coffee", the background scenes in the query-time dataset (bottom-right) are completely different from the test-time personalization dataset (bottom-left) depicting the same named instance.

## Dataset Overview

In This-Is-My dataset, we provide video segments and original videos for both the training and evaluation sets, along with annotated segments and captions for contextualized retrieval evaluation. The dataset structure is as follows:

```
<THISISMY_ROOT>/
    ├── train_segment/
    │   └── <SEGMENT_ID>.mp4, ...
    ├── eval_segment/
    │   └── <SEGMENT_ID>.mp4, ...
    ├── train_video/
    │   └── <{VIDEO_ID}_{VIDEO_NAME}>.mp4, ...
    ├── eval_video/
    │   └── <{VIDEO_ID}_{VIDEO_NAME}>.mp4, ...
    │
    └── this-is-my-dataset/
        ├── <SEGMENT>.csv
        ├── <TEST-SET>.json
        └── <EVAL-CAPTIONS>.csv
```

### Get Started
To get started, we recommend creating a conda environment and installing the required packages using the following commands:
```
conda create --name this-is-my python=3.7
conda activate this-is-my
conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch

# packages for downloading video segments
conda install -c conda-forge pytube
conda install -c conda-forge moviepy
conda install -c anaconda pandas
```

### How to Download the *This-Is-My* Dataset?
We have provided a simple script to download the dataset from scratch. Run the following command to download the video segments and original videos:
```
python download_video.py --MODE 'train'
```
This script will create two folders: train_segment\, which contains the video segments of named instances, and train_video\, which contains the original videos. 

Note that you can replace 'train' with 'eval' to download video segments for evaluation as well.

### Retrieving *This-Is-My* Metadata
```
python thisismy_dataset.py
```
We can retrieve the metadata of dataset by `load_thisismy(ANNO_FILE,SEGMENT_FILE)`. The returned variables contain the following information:

- `train_x`, `eval_x`:  Arrays that include segment IDs for the train and evaluation data splits (e.g., ead408e4-e1b6-4256-9adf-043906a41170)

- `train_y`, `eval_y`: Arrays that include token IDs (e.g., 0) for each segment. The token IDs can be mapped to instances using the `token2item` dictionary (e.g., {0: "Casey's friend marlan"})

- `train_class`, `eval_class`: array that includes category IDs (e.g., 7) for each segment. The category IDs could be mapped to category name by `id2classname` dictionary (e.g., {7: 'man', 8: 'piano'})

- `token2class`: A dictionary that provides a hierarchical mapping between token IDs and category IDs. (e.g., {0: 7, 1: 7, 2: 10, 3: 0})

We can also retrieve annotated data of eval captions by `load_this_is_my_captions(CAPTIONS_FILE)`, some returned variables are:

- `captions`: Annotated captions that describe the concept in the segment. (e.g., * is standing at the intersection)

- `class_names`: The class of the named instance (e.g., man)

## How to Get Support?
If you have any general questions or need support, please feel free to contact: [Chun-Hsiao Yeh](mailto:daniel_yeh@berkeley.edu), [Simon Jenni](mailto:jenni@adobe.com), and [Fabian Caba](mailto:caba@adobe.com). Also, we encourage you to open an issue in the GitHub repository. By doing so, you not only receive support but also contribute to the collective knowledge base for others who may have similar inquiries.


## Citation
If you find the This-Is-My dataset valuable and utilize it in your work, we kindly request that you consider giving our GitHub repository a ⭐ and citing our paper.

```bibteX
@inproceedings{yeh2023meta,
  title={Meta-Personalizing Vision-Language Models To Find Named Instances in Video},
  author={Yeh, Chun-Hsiao and Russell, Bryan and Sivic, Josef and Heilbron, Fabian Caba and Jenni, Simon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19123--19132},
  year={2023}
}
```



