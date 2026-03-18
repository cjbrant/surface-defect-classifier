# Results Summary

## Overall Metrics

| Metric | Value |
| --- | ---: |
| Validation samples | 854 |
| Accuracy | 0.9496 |
| Macro precision | 0.9579 |
| Macro recall | 0.9471 |
| Macro F1 | 0.9481 |
| Weighted F1 | 0.9482 |

## Class Level Performance

| Class | Support | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| Crazing | 162 | 0.8308 | 1.0000 | 0.9076 |
| Inclusion | 159 | 0.9813 | 0.9874 | 0.9843 |
| Patches | 193 | 0.9895 | 0.9793 | 0.9844 |
| Pitted surface | 87 | 0.9663 | 0.9885 | 0.9773 |
| Rolled-in scale | 132 | 0.9796 | 0.7273 | 0.8348 |
| Scratches | 121 | 1.0000 | 1.0000 | 1.0000 |

## Error Pattern

| True class | Main confusion |
| --- | --- |
| Rolled-in scale | 32 validation chips predicted as crazing |
| Inclusion | Minor confusion with crazing and patches |
| Patches | Minor confusion with inclusion and rolled-in scale |

Interpretation:

- Performance is strong overall, but rolled-in scale is the primary recall bottleneck.
- The confusion pattern suggests texture overlap between rolled-in scale and crazing at the chip level.
