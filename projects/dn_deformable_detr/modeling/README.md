# major version 1
## minor version 1.0
1. only use matching queries to generate cross-attention-weight map
2. extend the original reference points for encoder to reference boxes.
3. make use of attention weight threshold to identify background tokens which have tenuous, if any, links to objects. Their refence points remain as it own position.
4. tokens having potential objects use their corresponding box predictions as their new reference boxes.

## minor version 1.1
based on v1.0, we add:
1. only matching queries with cls > 0.2 contribute to the cross-attention map.


# major version 2
branch name: multiDA
## v2.0.0
In this version, we would like to validate the following idea:
Let object tokens explicitly focus multi objects.