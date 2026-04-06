Please catch up by reading associated documentation, previous logs, and this file
/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/claude-team/specs/motion-bert-round-4.md

## Phase 0: Tidying up the code

Yup let's omit the head entirely including in the visualization

Let's also simplify the bone scaling in motionbert_to_camera_space. This is way too complex for what we're trying to do. I'd prefer to focus on a single method, which is pairwise separation instead of also the torso height heuristic.


## Phase 1: Smooth motion over time

Please remove the ALL_JOINTS_SMOOTH_WEIGHT and everywhere it's used. Motion should be penalized via just the root position change and the rotation changes. Other joints' position changes will be penalized because smooth rotation implies smooth position change.

Hmm isn't there temporal smoothing via the smoothing factors during our optimization?

Just like in mediapipe where we solved pnp to get everything into CAMERA coordinates. I'd like you to do the same. It seems like you have another strategy (Please explain the strategy before you begin implementing) for estimating z values. But this should happen before our optimization such that our optimization only ever sees things in z coordinates and has full freedom to smooth out those values.

### Testing

Run on a super simple example. Have the tester use the following to ensure that the outputs are reasonable. 

1. pull out a frame from the heatmap video
2. Analyze data, especially from z values to ensure nothing looks completely off
3. MPJVE should be improved for ALL test videos. This is the purpose of smoothing itself.
4. Other metrics the tester is free to use

## Phase 2: Hyperparameter tuning 

Please further tune the hyperparameters (particularly smoothing and heatmaps blur)

Make sure the heatmap blurs you use are applied to the heatmaps and that those blurred heatmaps are the ones that show up in the heatmap overlay video.

Run on a multiple examples. 

At least one example should have a heatmap with a 5x radius because I'm curious to see what will happen in this scenario.

Remember to keep the issues we've been having in mind and have an eye out to see if they've been fixed or not.

Have the tester use the following to ensure that the outputs are reasonable. 

1. pull out a frame from the heatmap video and analyze it yourself
2. Analyze data, especially from z values to ensure nothing looks completely off
3. MPJVE should be improved for ALL test videos. This is the purpose of smoothing itself.
4. Other metrics the tester is free to use
