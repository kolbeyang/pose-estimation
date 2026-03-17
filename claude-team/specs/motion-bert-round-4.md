Alright there are several things that I notice looking at the results.

There are several questions here and I need answers. Please don't be afraid to reference actual code snippets, parts of previous documentation, or other references to make this easier for me to understand.

Make sure you're caught up on this information first and foremost.

/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/claude-team/specs/motion-bert-round-3.md

## Ground truth head keypoints

One is that the ground truth head and neck keypoints are way off. Is there some mapping that is causing this to be this wrong? I remember in mediapipe we were extending head to nose to estimate the top of the head. I'm worried we're doing something like that here. Instead in Mediapipe I remember that we decided to actually completely omit the top of the head and I think that might be the way to go here as well at least for now.

## Smooth motion over time

In the mediapipe example we were actually able to successfully smooth motion over time. It seems like we are failing to do that now. I make this conclusion from several observations.

First I'm looking at the z positions and they do not appear smooth at all. They are extremely jittery. I would have expected that the hip joint especially should have really smooth z values since that is the only point with actual position encoded. The other joints are all derived from angles (right?).

Second I'm looking at the per-joint MPJVE and we're actually doing worse than the raw motionbert outputs. This makes no sense to me.

## Motionbert implementation questions

1. Motionbert already does smoothing over time. Are we taking advantage of that or sending one frame at a time. It wouldn't be a fair comparison I think to send one at a time. We might need to discuss further on this one.

2. How are bone lengths being initialized for our optimization. I was surprised that motionbert's bones, our optimized bones, and the ground truth bones are so similar. I would have expected much worse due to the scaling issue we were experiencing.

3. How are we handling the scaling issue? We previously discussed scaling issues associated with being unable to recover the z-coordinate, Human 3.6m being performed with the 2.5d scale factor already known, and the pose keypoints being calculated relative to the root. In the visualization I was pleasantly but reluctantly surprised by how close the prediction, motionbert raw, and ground truth were along the z-axis. Describe this part of the pipeline in detail and explain any critical decisions that were made here in order to achieve this.

## Awful 2d MPJPE

Looking at the 2d MPJPE, some frames get up to 100px MPJPE. That's up to 10 percent of the total image height which seems unreasonably unrealistic. I looked at the videos and it doesn't look like we're a full 10% off anywhere.

I suspect this is either an issue with the calculation of MPJPE or maybe I misviewed these things and the video matches the graphs in actuality.


