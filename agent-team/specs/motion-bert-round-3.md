Please make sure you first reference ./motion-bert-round-2.md to get context on this task.

## Phase 0

Get a developer agent to do the following.

Get more CMU panoptic examples please and place them in the same place as the rest.

You can take a look here.
/Users/kolbeyang/Documents/School/spring_2026/capstone/panoptic-toolbox/


## Phase 1

Please get another developer agent to run all ten examples such that we get verbose output (all graphs and full videos with overlays) in parallel with Phase 0. 

## Phase 1: Parameter tuning

Full triple-sub-agent loop for this please.

Please try to get better performance. Another metric you should be tracking is the 2d-projected MPJPE measured in pixels. Project our prediction and the motionbert raw back down to 2d and measure the distances with the 2d prediction.

I'm noticing that both motion bert and our predicted 3d keypoints are failing to match the heatmaps well (they're doing okay) in the reprojections. This leads me to 2 hypothesis. 

1. Phase 1.1

The heatmaps strength is not strong enough relative to the other hyperparameters. Please do a coarse to fine approach to tuning the hyperparameters (order of magnitude up, order of magnitude down, 2 orders up, two orders down) to get a feel for how the parameters affect the output. 

2. Phase 1.2

The heatmaps are not wide enough to properly "capture" the points during gradient descent. Once you have found hyperparameters that work for Phase 1.1, let's play an extra blur parameter which we will use to blur the heatmaps that come out of Stacked Hourglass. This will make our system a bit more lenient, and give more signal for gradient descent to "grab on" to.

### Testing

Use MPJPE, P-MPJPE, MPVPE (velocity error), and 2d reprojection MPJPE (pixels).

Report your results at the end and make a recommendation with how to proceed.
