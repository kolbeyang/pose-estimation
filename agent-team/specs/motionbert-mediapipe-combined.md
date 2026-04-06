I want to create the final codebase. 

This should combine the functionality of the `mediapipe-pose` sub-directory and functionality of the `motionbert-pose` sub-directory into a single new subdirectory called `pose-optimizer`. Leave the original two subdirectories untouched.

There is quite a bit omitted from this spec, and it is expected that you are already familiar with the original `motionbert-pose` and `mediapipe-pose` implementations.

If there is a discrepancy between the way we do something in MediaPipe vs MotionBert, I'd prefer you bias towards using the method in MotionBert just because I've worked on that one more recently.

## File structure

The files and directories listed below are by no means exhaustive. This is a rough outline of the files required for the desired functionality.

```
data/                               Where the CMU data should live, gitignored please
cmu_data.py                         Any logic to download the CMU Panoptic data, or any shared utilities to parse the CMU Panoptic data
motionbert/                         
    __init__.py                     The Motionbert entrypoint
    run_single_config.json          Config to run just a single example
    run_full_config.json            Config to run on all datapoints
    ...                             Related Motionbert files
mediapipe/                         
    __init__.py                     The Mediapipe entrypoint
    run_single_config.json          Config to run just a single example
    run_full_config.json            Config to run on all datapoints
    ...                             Related Mediapipe files
optimize/                           Our optimizer
    __init__.py
    ...                             Related optimizer files and utilities
experiment/                         Catch-all folder for any experiments the agent needs to do during development or testing
    ...                             Any one-off test scripts or even more reusable things can be placed here
visualize.py                        Visualization script
output/                             All output should default to here
    motionbert_03_28_04_45/             Directory per output denoted with the timestamp in the name
        video_1/                    Directory per video
            summary.png             All important graphs appended into a single summary image (same as we do already)
            overlay_video.png       Overlay video (same as we do already)
            results.json            Evaluation metrics (described elsewhere)
            graphs/                 Full graphs as individual images
                ...
            trajectories.json       Trajectory data for ground truth, raw prediction, optimized prediction which will be input to the visualize script
        video_2/
            ...
    mediapipe_03_28_05_15/
        ...
```

## Data
You will see some examples used in the `motionbert-pose` and `mediapipe-pose` implementations. Here let's use the dedicated `data` folder to store the examples and associated data.

Write the necessary data downloading functionality and any other associated helpers into the `cmu_data.py` directory.

The current implementations use about 10 examples or so for the full evaluations. For our purposes let's increase that to about 25 examples for our full runs. Obviously during small tests we should only be using single examples. 

Let's only use examples with just one subject if it's possible to filter them that way anyhow.

## Base 3d Pose Estimators

### Mediapipe
The Mediapipe entrypoint should be runnable via `uv run mediapipe.py config.json` 

This should run the full pipeline `Mediapipe` -> `Synthetic heatmaps` -> `Optimizer`

The synthetic heatmaps used in the Mediapipe example are just a Gaussian applied to the 2d landmarks. The sigma of the Gaussian should be a configuration option for Motionbert.

### Motionbert
The Motionbert entrypoint should be runnable via `uv run motionbert.py config.json` 

This should run the full pipeline `YOLO` -> `StackedHourglass` -> `MotionBert` -> `Optimizer`

### API

The Motionbert and Mediapipe files should behave with the same API. 

No need for `--graph` or `--visualize` for these scripts although they exist in the original.

INPUT:
- config.json
    - Should include paths to videos, camera parameters, ground truth, etc. that will be processed
    - Output directory: optionally set an output directory which will be created if not existing already. Default to our timestamp based output folder structure if not specified.
    - Optimzer params, learning rates, heatmap blur, heatmap sigmas, motion and rotation penalties
OUTPUT:
- Graphs, video, evaluation metrics, etc. as described above in the output directory

## Camera Object

The camera object will be responsible for owning the camera parameters and projection logic.

It should handle any functionality like `world_to_camera`, `camera_to_world`, `camera_to_image`.

Handling this in a single abstraction allows us to pass this around easily wherever it's needed.

## Skeleton model

These models are use slightly different skeleton models with different numbers of keypoints. I'm pretty sure a 16 point skeleton is the "lowest-common-denominator" so to speak of all these models. Do not extrapolate points by, for example, doubling the distance between the neck and the nose to get the head. This yields unexpected results and unneeded inaccuracies.

These mapping decisions are critical. I trust your judgement here but these decisions are ones you should highlight in your final report.

## Shared Optimizer

The idea here is for the optimizer to be more of a pure function:
INPUT: 2d heatmaps, raw 3d predictions, Camera object
OUTPUT: improved 3d predictions

## Evaluation Metrics

Evaluation metrics should live in an `evaluate.py`:

Here's an explanation of some of the modifiers that are being applied to the metrics below.

**SI** - Scale Independent
This has already been used in part in the current iteration of `motionbert-pose`. Since single camera pose estimation has almost no signal to predict the scale factor, we can remove it from evaluation by finding an optimal scalar s to multiply the entire trajectory by before evaluating it. Please note this scalar is shared among all frames in the trajectory, not optimized per-frame.

**VW** - Visibility weighted
Some keypoints are not in frame and should not be included in the evaluation. Currently in the current iteration of `motionbert-pose` we weight the key points by their StackedHourglass confidence scores. However we are changing the evaluation to evaluate based on the frame boundaries.

Here is some loose pseudocode to demonstrate what I mean, do not copy verbatim. 
```
visibility_weight = current_frame.in_frame_bounds(ground_truth_3d_coord.project_to_2d(...camera, ...other necessary arguments))
```
These visibility weights should be calculated per key point per frame and applied accordingly.

### MPJPE: Mean Per-Joint Pose Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
**OUTPUT**: 
- Score

### P-MPJPE: Procrustes Mean Per-Joint Pose Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
**OUTPUT**: 
- Score

### SI-MPJPE: Scale-Independent Mean Per-Joint Pose Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
**OUTPUT**: 
- Score

### VW-MPJPE: View-Weighted Mean Per-Joint Pose Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
- Keypoint visibility scores: (N, K)
**OUTPUT**: 
- Score

### VW-SI-MPJPE: View-Weighted Scale-Independent Mean Per-Joint Pose Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
- Keypoint visibility scores: (N, K)
**OUTPUT**: 
- Score

This is the primary evaluation metric that should take some precedence over the others.

### MPJVE: Mean Per-Joint Velocity Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
- Keypoint visibility scores: (N, K)
**OUTPUT**: 
- Score

### SI-MPJVE: Mean Per-Joint Velocity Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
- Keypoint visibility scores: (N, K)
**OUTPUT**: 
- Score

### VW-MPJVE: Mean Per-Joint Velocity Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
- Keypoint visibility scores: (N, K)
**OUTPUT**: 
- Score

### VW-SI-MPJVE: Mean Per-Joint Velocity Error
**INPUT**: 
- Predicted trajectory: (N, K, 3)
- Ground truth trajectory: (N, K, 3)
- Keypoint visibility scores: (N, K)
**OUTPUT**: 
- Score

Pay attention to this metric, since we have smoothing terms, our optimization should be outperforming the raw output from both `motionbert` and `mediapipe`.

## Testing

The exact testing methodology required will vary based on the exact phase of development you are working in, but here are some general guidelines.

### Time investment
An important principle to adhere to is 
```
Time investment should be proportional to confidence it will work
```
This means at the beginning, run very very rough tests. As your confidence grows in a particular piece of functionality, it is okay to run on 3 examples, 5 examples, 10 examples etc.

### Heatmap videos
The heatmap videos generated with each run by default provide strong signal as to what is happening within the models.

In order to effectively analyze them, use pillow or ffmpeg to extract a frame from midway through the video. Analyze this still image and particularly analyze how the following elements are placed relative to one another:
- The actual human subject of the video
- The heatmap overlays
- The ground truth skeleton
- The raw base model skeleton
- The optimizer's predicted skeleton

### Evaluation metrics
Use the evaluation metrics to judge the quality of the current run.

