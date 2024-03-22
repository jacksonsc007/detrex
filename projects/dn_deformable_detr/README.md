branch name: coarse-to-fine

In this branch, we input feature maps discriminately to each *cascade_stage* to form a coarse-to-fine detection pipeline.

# v1.0.1
In this minor version, we assign less weight to shallower stages by add scalar to the original weight coefficients.

For a 3-layer model:

|layer_idx|scalar| 
|-|-|
|0|0.333|
|1|0.666|
|2|1|


> Note that we modify the config of model. Coco-minitrain is now not supported.