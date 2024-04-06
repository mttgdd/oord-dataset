# OORD: The Oxford Offroad Radar Dataset

Software development kit and experiments for [The Oxford Offroad Radar (OORD) Dataset](https://oxford-robotics-institute.github.io/oord-dataset/).

Please cite the following <a href="https://arxiv.org/pdf/2403.02845.pdf">paper</a>

**OORD: The Oxford Offroad Radar Dataset** <br>
M. Gadd, D. De Martini, O. Bartlett, P. Murcutt, M. Towlson, M. Widojo, V. Muşat, L. Robinson, E. Panagiotaki, G. Pramatarov, M. A. Kühn, L. Marchegiani, P. Newman, L. Kunze<br>
<i>arXiv preprint arXiv:2403.02845</i>, 2024 <br>

```bash
@article{gadd2024oord,
title={{OORD: The Oxford Offroad Radar Dataset}},
author={Gadd, Matthew and De Martini, Daniele and Bartlett, Oliver and Murcutt, Paul and Towlson, Matt and Widojo, Matthew and Mu\cb{s}at, Valentina and Robinson, Luke and Panagiotaki, Efimia and Pramatarov, Georgi and K\"uhn, Marc Alexander and Marchegiani, Letizia and Newman, Paul and Kunze, Lars},
journal={arXiv preprint arXiv:2403.02845},
year={2024}
}
```

# Notes

1. As per the [Oxford Radar RobotCar Dataset](https://arxiv.org/pdf/1909.01300.pdf), there are a very small number of data packets carrying azimuth returns that are infrequently dropped. However, our `dataset.py` script resizes both polar and cartesian arrays to a fixed size.
