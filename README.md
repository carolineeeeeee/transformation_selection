# What to Check: Systematic Selection of Transformations for Analyzing Reliability of Machine Vision Components

## Required Inputs
Files are stored in inputs/
1. a csv file containing image transformation information, called *transformations.csv*. The script *gen_transf_csv.py* is used to generate this csv. **Note**: only modify *gen_transf_csv.py* to add or remove transformations. Currently, it contains all the transformations in the library [Albumentation](https://albumentations.ai/docs/getting_started/transforms_and_targets/) and library [Torchvision](https://pytorch.org/vision/stable/transforms.html).
2. a csv file containg CV-HAZOP entries. E.g., *cv_hazop_all.csv* contains all CV-HAZOP entries, and exp_extries contains 20 entries used in our evaluation experiment. 

The required inputs should be placed in the same directory as main.py.

## Instructions for Running
### File to run: *main.py*

`python main.py -c name_of_entries_csv -t name_of_transformation_library`

This file matches given CV-HAZOP entries to the give list of image transfomrations. 

For example, to match the entire CV-HAZOP checklist with image transformations in Albumentation, the command is:

`python main.py -c cv_hazop_all -t albumentations`

**Note**:
1. All transformations should be stored in *transformations.csv*
2. for the csv file containing CV-HAZOP entries, only include the name without '.csv' extension. For example, to consider *cv_hazop_all.csv*, only use 'cv_hazop_all'
3. The library name should exist in *transformations.csv*.
4. Python 3.8.5 is recommended.

## Outputs
We also included saved pickle files for evaluation results in outputs/. 
1. *cv_hazop_all.pickle*: contains all effects and actions for all entries
2. *albumentation.pickle*: contains all effects and actions for all transformations in the library [Albumentation](https://albumentations.ai/docs/getting_started/transforms_and_targets/)
3. *torchvision.pickle*: contains all effects and actions for all transformations in the library [Torchvision](https://pytorch.org/vision/stable/transforms.html)
4. *albumentation_eval.pickle*: contains the matching of scene changes in all CV-HAZOP entries to Albumentation image transformations. 
5. *torchvision_eval.pickle*: contains the matching of scene changes in all CV-HAZOP entries to Torchvision image transformations. 

## Experiment with experts
*expert_results.pdf* contains: 

1. Individual expert results for identifying image transformations simulating the CV-HAZOP entries. 

2. 'Ground truth' obtained by transformations agreed by the majority of the experts participated in the experiment after the discussion of difference in their results and our results. 

3. Mapping results by our systematic method and the automation of the systematic method. 

4. Challenges identified by each participant and their feedback on how well our method addressed it. 


## Extra RQ2 evaluation results
We also checked how well the image transformation library [Torchvision](https://pytorch.org/vision/stable/transforms.html) (51 transformations in total) covers the hazardous situations in the CV-HAZOP checklist.


| CV-HAZOP Location | situation-coverage | scene-change-coverage|
| ----------------- | ------------------ | -------------------- |
| Light Sources     |    0.0% (0/122)    |10.9% (29/266)        |
| Medium            |    0.0% (0/79)     | 23.2% (44/190)       |
| Object            |  1.9% (3/162)      |12.0% (37/308)        |
| Objects           |  0.0% (0/178)      |4.2% (14/334)         |
| Observer - Optomechanics| 0.0% (0/253) | 13.5% (69/512)       |
| Observer - Electronics |4.0% (5/125)   | 22.8% (55/241)       |

Coverage metric for different parameters at Observer - Electronics

| CV-HAZOP Parameter | situation-coverage | scene-change-coverage|
| ----------------- | ------------------ | -------------------- |
| Exposure and shutter    |   4.2% (1/24)    |14.3% (7/49)        |
| Resolution (spatial)    |    7.4% (2/27)     | 22.4% (13/58)       |
| Spectral efficiency    |  4.3% (1/23)     |31.0% (13/42)   |
| Quality           |  4.8% (1/21)      |23.1% (9/39)       |
| Quantization/Sampling| 0.0% (0/30 | 24.1% (14/58) |