The main development branch for this work should be `feat/motionbert-optimization`. Create this branch if not created already.

# Phase 1: Tracking limb length over time

Please include a graph of the following limb lengths over time:
- Right elbow -> right wrist
- Right shoulder -> right elbow
- Left elbow -> left wrist
- Left shoulder -> left elbow

Do this for:
- Ground truth (blue)
- Motionbert raw 3d predictions (green)
- Optimized 3d predictions (red)

 Resulting in a total of 3 * 4 = 12 lines in the line graph.

## Testing

Sanity check the graph itself to make sure it makes sense.

# Phase 2: Improving handling on occlusion

First let's remove the function in `motionbert-pose/scoring.py` named `heatmap_score`. We never want to do synthetic heatmaps.

The function called `real_heatmap_score` should then be renamed to just `heatmap_score`.

Next I'm really focused on this line in that function.
```
log_val: torch.Tensor = torch.log(torch.clamp(value, min=eps))
```

I would like this scoring function to also take into account the confidence scores that are coming from StackedHourglass and it should be modified to match the following pseudocode:
```
log_val = log( value * confidence + confidence_epsilon * ( 1 - confidence ))
```

This format specifically is to preserve our probabilistic interpretation of the data.

Currently, I suspect there is some logic to omit occluded keypoints from scoring altogether but that logic should be removed in favor of this logic which is much cleaner and is more mathematically rigorous.

## Visualization

### Confidence Score Graph

I would like to see a graph being produced in the runs folders and in the summary view that shows me how visible each of the key points is. 

Please make a line graph where each line is one of the keypoints across the frames in the video.
x-axis: frame of the video
y-axis: StackedHourglass confidence score

### Proper thresholding for overlay videos

The issue right now is that keypoints that are obviously not in the video are predicted in places that are completely and obviously off. 

I want you to select a threshold at which it would be reasonable to omit that keypoint from the heatmap overlay video visualization, and from the 3d visualization.

In order to do this, run one of the example videos and either print the confidence values over the frames or interpret the graph that you will make in the previous section. 

Then save out one of the frames and verify which points are clearly not in frame and cross reference with the corresponding confidence scores. This should give you a rough idea of where a threshold might be reasonably placed.

## Testing

### Occluded test videos
I want you to select 2 of our existing CMU panoptic test videos and augment them in the following ways:

1. Every other frame is black (Pattern = [visible, black])
2. Every fourth frame is visible (Pattern = [visible, black, black, black])
3. Every fourth frame is black (Pattern = [visible, visible, visible, black])
4. 16 frames visible, 16 frames black (Pattern = [visible * 16, black * 16])

# Phase 2: Trajectory-level P-MPJPE

Currently when we evaluate P-MPJPE, we basically give each frame invariance to scale rotation and translation.

This is far too generous.

We should create a new metric called Scale-Z-Invariant-MPJPE. The only dimension which our model truly cannot estimate is scale/Z. Meaning that from a single camera, we can estimate everything else except a single scale factor. We cannot tell if something is big and far away or small and close.

We need to find the scale, s, which minimizes the MPJPE. 

If we write this out and set the derivative to 0, we end up getting:

MPJPE = arg min of s 1/(num_keypoints)(num_frame) sum_over_all_keypoints_and_all_frames(squared_magnitude(scale * predicted_point - gt_point))

It ends up coming out to this

s = (sum_over_all_keypoints_and_all_frames(predicted_point * gt_point) / sum_over_all_keypoints_and_all_frames(gt_point * gt_point))

Make sure this algorithm works and make sense. Apply this scale to whatever trajectory we're measuring and then calculate the MPJPE as usual by comparing to ground truth.

## Testing

Write some temporary unit tests to make sure this algorithm is working as expected. You can even render an image of the 3d skeleton before multiplying by the scale factor, after multiplying by the scale factor, and the ground truth. Make sure the image is from the perspective not near the camera and ensure that the scaled figure is closer to the ground truth than the original figure is.

# Phase 3: Performance and latency reduction

## Benchmarking

In order for this to be effective we need accurate benchmarks, which will tell us if our system is improving or not. 

The accuracy metrics in order of importance are
1. MPJPE
2. P-MPJPE
3. MPJVE

The latency metrics in order of importance are
1. total end-to-end latency / frames of input

Find these values for all of our current test input videos and save them.

You must also save out a data file with the MPJPE throughout training, you'll see why later.

## Profiling

Find a tool for profiling and keep track of how long each step takes, use this info later to assess whether your improvements have made any progress.

## Experiments

You will run a variety of experiments in an attempt to try an improve the performance of the system. Come up with a detailed experimentation plan based on my suggestions. This plan can evolve as you make new discoveries along the way.

Here are some approaches that you can consider taking.

1. Reduce iterations and increase learning rate
2. Reduce time per iteration
3. Increase vectorization and parallelization 
4. Identify slow-running steps and optimize them

All of the hyper-parameters are up for grabs as well as you may find that at different learning-rates, different penalty coefficients perform better, or something like that.

## Experimental methodology

Training time is precious and your experimentation should follow a tiered approach.

When testing a new configuration, run it only for a few iterations and confirm that compared to the baseline or the current best configuration, that it is performing reasonably well. If it blew up already in the first few iterations, it is not worth spending more time on.

These quick smoke tests do not need to be saved at all.

Then increase the iterations by an order of magnitude or so, and see how it performs there.

The main principle you need to keep in mind is
```
Time investment into a hypothesis must be proportional to the evidence that it is going to work.
```

Runs that you think are significant enough should be saved in git history. Create a new branch at that point and switch back to the main development branch to continue work.

## Visualization and reporting

At the end, I want to see all of your runs on a graph with:
x-axis: training-time (seconds)
y-axis: MPJPE

I would also like to see other graphs that you determine useful for me to evaluate the experiments you've run. The graph above should give you an idea of what I'm looking for.

Make sure each datapoint is labelled with consistent descriptive names.

Save out a markdown file with a standardized YAML header detailing each experiment. These markdown file names should match the datapoint labels.

In your final report, describe your top choice for configuration, make sure that is the version that is left "on the desk" meaning that when I come back and run `uv run main.py` that is the configuration that will run.

Explain which changes were unequivocally improvements and which changes involved tradeoffs.

# Agent Orchestration

The testing sections of this spec should be delegated to the testing agent.

## Reporting
When I come back, I want to catch up on all important context quickly. In order to achieve this, I need to know what worked, what failed, and any critical decisions that were made.

Instruct sub-agents after each run to report the following:
- Things in the environment or issues with the spec itself which prevented them from doing their job properly
- Critical decisions that they made which may need my review later
- Anywhere where the spec was unclear and they made some inferences or leaps which may be wrong

In your final report to me I want you to give me all of the sub-agent feedback and also include a section on how I can improve these spec files to get things running more efficiently in the future.
