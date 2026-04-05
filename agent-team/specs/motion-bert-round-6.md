Please read this and get caught up
claude-team/specs/motion-bert-round-5.md

Then let's discuss rotation penalties. 

You mentioned some type issues, just spin up the devloper to fix that real quick.

Then

Essentially I want you to tune hyperparameters and find the best rotation_penalties

Develop a moderately robust searching plan with minimal code changes to find the best rotation. You can play with tuning individual rotation params, but better to just scale them all up 10x down 10x up 2x down 2x etc.

Only run the architect to come up with the plan.

Then the tester can take care of the test.

Make sure they "As you complete the task you may make some critical decisions that I may want some insight into. You may also flag code that you see that may cause issues in the future that would also demand my attention. At the end of your implementation, come back to me and tell me about any and all critical decisions that you made so that I can review them."
