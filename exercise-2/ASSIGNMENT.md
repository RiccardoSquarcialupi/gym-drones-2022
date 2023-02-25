Assignment 2
The assignment consist in two squad of enemy drones (blue and red team), each of them composed by 4 drones.

The spawn position of each drone is fixed for the two squads.
- accordingly:
        -  Red team:
            -  1: -0.5 + (0.5 * 1), 5, 2
            -  2: -0.5 + (0.5 * 2), 5, 2
            -  3: -0.5 + (0.5 * 3), 5, 2
            -  4: -0.5 + (0.5 * 4), 5, 2
        -  Blu team:
            -  1: -0.5 + (0.5 * 1), -5, 2
            -  2: -0.5 + (0.5 * 2), -5, 2
            -  3: -0.5 + (0.5 * 3), -5, 2
            -  4: -0.5 + (0.5 * 4), -5, 2
According to the code done in class.

Each drone must face the enemy counterpart at spawn time.
Each drone has 5 spheres that he can shoot, with a maximum rate of fire of 1 per seconds


There are no restriction on how to shoot the spheres, the Team can autonomously decide the technique, but speed and scaling of the sphere are fixed at:
- scaling: 2
- approx speed: 50
- urdf of the sphere: sphere_small.urdf


The policy can use any information of the drone that it drive and any information from the other drones of the same team.

The battle can begin after 3 seconds of flight in simulation (try to avoid spawn kill), this mean that any attempt to shoot before 3 sec must be punished and lead to no ball fired.

Be careful that this mean that each policy must be aware of the time from the start of the simulation, the elapsed time must be counted keepening the number of elapsed steps and the TIMESTEP of the simulation.
Change the frequency of the simulation is not allowed.

The objective of the assignment is to train multiple policy that drive and lead one team to victory.

Obviously, one drone dies if it collides with anything, like walls, spheres and other drones.

Since this is a very difficult task, the evaluation will not only be based on the performances achieved, it will mainly be based on the choices and techniques adopted for training.

I have tried the [SquarciaLupi porting rlib 2.0](https://github.com/RiccardoSquarcialupi/gym-drones-2022.git)
 and seems to be working correctly, starting from this code also, I have resolved the issue of the termination.
Still, the ray 2.0 migration is not yet done, but I can confirm that it works.

The issue was quite deep, and hard to fully comprehend.

The issue was inside the computeObs method that always return NUM_DRONES observation, this cause a misleading behavior by Rlib that after a dead drone remove the corresponding action from the action given to the next step, the problem is that following [Ray doc](https://docs.ray.io/en/latest/rllib/rllib-env.html) documentation we can see that doing so will revive our killed drone after only one step.

> When implementing your own MultiAgentEnv, note that you should only return those agent IDs in an observation dict, for which you expect to receive actions in the next call to step().

If drone 0 is dead in the step 100, and we revive it in the step 101 returning his drone ID in the dict of the computeObs then the bug occurs.

The original code was not designed to handle agent that can die earlier, so I have done some modification to the code of Squarcialupi to try to resolve the bug (Mb the Rlib 2.0 resolve the problem, but no), I have tried to keep as much attention to the part where the code need to be changed to make it work properly, this does not mean that the code is fully functional and there are no bugs around.

The fact that the previous code uses the index of the drones (using something like: for i in range(NUM_DRONES)),  this lead to a new problem: if we have 3 drones and the corresponding index are 0,1,2 then when the drone 1 dies the drone 2 got the position of the drone 1 swapping the index, this is caused by the fact that the code does not rely on the IDs of the actuals drones alive but only on the starting defined number NUM_DRONES, i have tried to correct this behavior using the Rlib variable  _agent_ids where needed, and dynamically return the correct number of observation, keeping the action for the dead drone to [0,0,0,0] (so that dead drones fall to the ground).


I know that these assignments are kinda messy, due to many reasons, I really apologize for this.
Keep it up, I remain to your disposal when needed.
If you need a meeting, we can agree on a common date for the interested parties.


Maybe we can try to create a podium between the teams, we'll see how it goes and then decide.


For the Assignament2, provide the repository used, that contains user code.

If something to you is missing from the assignment description,  let me know as fast as possible,  so that all teams can be aligned.

As the deadline, I propose two months from now, but it can be changed if needed.

I think that all team should cooperate, still keeping their like ideas original as much as possible, but this task is hard and if you all collaborate maybe you can also finish the assignment faster, and with less difficulty.

Assignment 2 repo: https://github.com/camillo2008/gym-pybullet-drones/tree/ass2