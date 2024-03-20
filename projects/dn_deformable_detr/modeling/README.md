branch: ref-init

In this branch, the initialization of sampling locations is researched.

# v1.0
we initialize the weight and bias of sampling offset as all 0.

Results show that the initialization sampling offset matters. 

# v1.2
In this version, we expand reference points to reference boxes for encoder.

# v1.3
on the basisi on  v1.2, we limit the sampling points are within the range of box.


# v1.3.1
1. we increase the number of sampling points.
2. fix tanh bug and use sigmoid
3. change softmax range from object level to head level.


# v1.3.2
1. we modify the initialization of bias of the sampling_offset of multi_scale_deformable_attention.