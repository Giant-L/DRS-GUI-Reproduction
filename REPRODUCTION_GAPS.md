# Reproduction Gaps

This document records DRS-GUI details that are not fully specified in the supplied paper. No DRS-GUI algorithm is implemented on the Stage 1 `main` branch.

## Known questions for Stage 2

### Focus spatial outlier removal

- **Paper description:** remove elements whose centers deviate markedly from the cluster centroid.
- **Missing detail:** distance metric and outlier threshold.
- **Our assumption:** not selected yet.
- **Possible impact:** changes which elements define the focused crop.

### Focus target shrink ratio

- **Paper description:** iteratively prune farthest elements until a target shrink ratio is satisfied.
- **Missing detail:** target ratio and stopping/tie behavior.
- **Our assumption:** not selected yet.
- **Possible impact:** directly controls crop size and target recall.

### Shift directional grouping

- **Paper description:** group external anchors by a consistent layout direction.
- **Missing detail:** direction bins, grouping rule, and candidate-region construction.
- **Our assumption:** not selected yet.
- **Possible impact:** changes Shift proposals and MCTS branching.

### Reward constants

- **Paper description:** non-interactive weight `lambda`, semantic temperature `tau`, and numerical `epsilon` appear in the equations.
- **Missing detail:** their numerical values are absent from the supplied main paper.
- **Our assumption:** not selected yet.
- **Possible impact:** changes region ranking.

### Domain-specific prefix

- **Paper description:** construct a prefix from application and system type before instructor-large encoding.
- **Missing detail:** exact template text.
- **Our assumption:** not selected yet.
- **Possible impact:** changes semantic relevance scores.

### MCTS operational details

- **Paper description:** initial Focus, UCT selection, one unexecuted action expansion, reward evaluation, and backpropagation with `N=8`, `H=3`, `c=1`.
- **Missing detail:** no-op handling, duplicate-region handling, tie breaking, Q-value update convention, and whether UI parsing is recomputed per crop or filtered from a global parse.
- **Our assumption:** not selected yet.
- **Possible impact:** changes explored trees and final best region.
