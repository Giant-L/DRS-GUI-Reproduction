# Reproduction Gaps

This file separates paper-specified behavior from implementation assumptions on the `reproduction` branch. The supplied PDF and the authors' arXiv source archive were checked. No official DRS-GUI code or supplementary parameter file was found as of 2026-09-11.

## Focus spatial outlier removal

- **Paper description:** remove selected elements whose centers deviate markedly from the cluster centroid.
- **Missing detail:** distance definition and threshold.
- **Our assumption:** Euclidean center distance greater than `0.55 * current_region_diagonal` is an outlier. If this removes every element, retain the highest-relevance element.
- **Reason:** deterministic, resolution-independent, and configurable through `focus_outlier_distance_fraction`. The original reimplementation used `0.35`; a fixed 10-sample search-only development diagnostic showed that this contributed to over-contraction. A more permissive `0.45`/`0.90` configuration improved search recall on a disjoint diagnostic group but reduced final grounding accuracy, so it was rejected rather than reported as an end-to-end improvement.
- **Possible impact:** directly changes target recall and the initial Focus region. The 0.55 value is a calibrated reproduction assumption, not a paper default.

## Focus target shrink ratio

- **Paper description:** if the minimal crop does not shrink sufficiently, iteratively remove the farthest element until a target shrink ratio is reached.
- **Missing detail:** ratio, area-vs-side-length interpretation, and tie handling.
- **Our assumption:** require crop area `<= 0.80 * previous_region_area`; recompute the centroid and remove the farthest element, breaking ties toward lower relevance and then stable element ID.
- **Reason:** area is consistent with the paper's image-area reduction analysis and keeps the rule deterministic. The original `0.60` assumption compounded across repeated Focus actions and produced a mean final area far below the paper's reported approximately 36% of the original image. The selected `0.80` balances recall and crop compactness on the development diagnostic instead of maximizing recall alone.
- **Possible impact:** the less aggressive threshold improves recall but can retain more clutter. It is a calibrated reproduction assumption, not a paper default.

## Percentage rounding

- **Paper description:** top 15% or top 10% elements.
- **Missing detail:** rounding and zero-count behavior.
- **Our assumption:** `ceil(count * fraction)`, with at least one element when the candidate set is non-empty.
- **Reason:** avoids disabling an action on sparse interfaces.
- **Possible impact:** small candidate sets may retain proportionally more elements.

## Shift directional grouping and region construction

- **Paper description:** group external anchors by consistent layout direction (for example upper or left), recenter around them, and minimize overlap; implementation specifies top 15% external anchors and IoU <= 0.3.
- **Missing detail:** direction bins, group selection, target dimensions, and boundary behavior.
- **Our assumption:** dominant-axis `left/right/up/down` groups; select the group containing the highest-relevance external anchor; preserve the current region width/height; move at least far enough along that axis for IoU <= 0.3; reject when screen clipping makes the constraint impossible.
- **Reason:** directly realizes the described recentering without silently changing field-of-view scale.
- **Possible impact:** diagonal layouts and boundary cases may differ from the authors' implementation.

## Scatter admission under the area cap

- **Paper description:** add top 10% external relevant elements while limiting region area expansion to 1.5x.
- **Missing detail:** behavior when an external element cannot be enclosed without violating the cap.
- **Our assumption:** consider external elements in descending relevance order and skip an element if its enclosing union would exceed 1.5x; return no proposal if none can be admitted.
- **Reason:** preserves both the inclusion semantics and hard area cap.
- **Possible impact:** a distant, highly relevant element may be skipped rather than included in a capped region.

## Reward constants and empty regions

- **Paper description:** `0 < lambda < 1`, temperature `tau`, numerical `epsilon`; no values are supplied. Equations 6-11 define the reward and final weights are 0.4/0.4/0.2.
- **Missing detail:** `lambda`, `tau`, `epsilon`, and the zero-element case.
- **Our assumption:** `lambda=0.5`, `tau=0.1`, `epsilon=1e-8`; all reward components are zero when the region has no visible elements.
- **Reason:** numerically stable, centralized defaults that make model-independent tests possible.
- **Possible impact:** especially `tau` can substantially change semantic concentration and region ranking. None of these three values is a paper default.

## Coverage for partially visible cached elements

- **Paper description:** `sum Area(b_i) / Area(R)` over elements parsed in region `R`.
- **Missing detail:** treatment of elements crossing the crop boundary when global parser output is reused.
- **Our assumption:** include elements whose centers lie in the region and use `Area(b_i intersect R)` in coverage.
- **Reason:** emulates boxes clipped by a parser operating on the crop and avoids counting pixels outside the region.
- **Possible impact:** differs if the authors re-run OmniParser per crop or use another visibility rule.

## Domain-specific prefix

- **Paper description:** prepend a prefix based on application and system type. Figure 2 shows `Represent the Word macOS UI element:`.
- **Missing detail:** general template, capitalization, separators, and whether instruction and description use distinct prompts.
- **Our assumption:** `Represent the {application} {platform} UI element: ` for both instruction and description.
- **Reason:** direct generalization of the only visible paper example.
- **Possible impact:** changes Instructor-large embeddings and all downstream relevance scores.

## UI parsing across regions

- **Paper description:** each state contains current region `R`, parsed elements `U`, and scores; actions refer to relevant elements outside the current region.
- **Missing detail:** whether OmniParser is re-run on every crop, whether a full-screen parse is retained for external anchors, and how local boxes are mapped.
- **Our assumption:** parse the full screenshot once, store all boxes in original pixels, and use center membership to derive visible/external elements at every state.
- **Reason:** supports external Shift/Scatter anchors, avoids repeated parser inference, and keeps core tests independent of OmniParser.
- **Possible impact:** crop-specific OCR/icon captioning can differ from filtering a global parse.

## MCTS operational details

- **Paper description:** post-Focus root, UCT selection, one unexecuted action expansion, immediate region reward, and backpropagation with `N=8`, `H=3`, `c=1`; choose the highest-reward region.
- **Missing detail:** whether initial Focus counts toward depth, action ordering, invalid/no-op actions, Q update convention, duplicate states, and ties.
- **Our assumption:** post-initial-Focus root has depth zero; deterministic order Focus/Shift/Scatter; invalid actions are marked attempted and the rollout tries the next action; Q is the arithmetic mean of backed-up candidate rewards; equivalent states reached by different paths are retained; final ties prefer smaller area then older node.
- **Reason:** deterministic, inspectable behavior matching the paper's rollout-free immediate reward description.
- **Possible impact:** changes tree shape and which regions receive the limited eight expansions.

## Real screenshot proxy demos

- **Paper description:** OmniParser V2 plus Instructor-large.
- **Missing component in current environment:** neither model is installed or executed.
- **Our assumption:** only for six explicitly labeled smoke demos, Tesseract OCR boxes and token-overlap scores were cached. All OCR boxes were marked interactive because the proxy cannot infer interaction state.
- **Reason:** verify real-image dimensions, action geometry, MCTS traces, and visualization without a model download.
- **Possible impact:** major and expected; proxy results are not paper-aligned and must never be reported as DRS-GUI accuracy.

## OmniParser V2 checkpoint revision

- **Paper description:** use OmniParser V2 as the UI parser.
- **Missing detail:** exact OmniParser source revision and detector checkpoint hash/version.
- **Our assumption:** source commit `354021201345a96178360b28733573e27269f2de`, with the V2 `icon_detect/model.pt` and `icon_caption` weights mirrored by `AI-ModelScope/OmniParser-v2.0`.
- **Reason:** the official Hugging Face host and its Xet blob endpoint timed out from the rented AutoDL instance, while ModelScope provided the original V2 file layout. The newer `icon_detect_v3` PR checkpoint was not substituted silently.
- **Possible impact:** detector boxes may differ from the authors' unpublished environment or the repository's later YOLOv9-E checkpoint.
