import pdb
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
import gym
import logging
import random
from gym import spaces
logging.basicConfig(filename='aviary.log')

GOAL_POSITION = [60, 0, 0]
WORLD_MARGIN = [[-20,60],[-10,10],[0,10]]

dpp = np.array([[0, 0,  0.1125],
                                [0.1588, 0.1588, 0.1125],
                                [0.3176, 0.3176, 0.1125],
                                [0.4764, 0.4764, 0.1125]])
num_resets = -1
env_number = 199
max_drones_states = [-13, -13, -13, -13]

class ReachThePointAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""
    ################################################################################
    global dpp
    global num_resets
    global env_number
    global max_drones_states
    global WORLD_MARGIN
    
    initial_xyzs = np.array([[-10 + random.uniform(-3, 3), random.uniform(-5, 5), 1] for _ in range(4)])

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X, ##drone model
                 num_drones: int = 2, #number of drones
                 neighbourhood_radius: float = np.inf, #ignore
                 initial_xyzs=initial_xyzs, #n-shaped array containing the initial XYZ position of the drones(n)
                 initial_rpys=None, #n-shaped array containing the initial orientations of the drones (in radians).
                 physics: Physics = Physics.PYB, #physics of env
                 freq: int = 240, #The frequency (Hz) at which the physics engine steps. 
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.last_drones_dist = initial_xyzs

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        import pybullet as p
        import csv
        import os
        from random import randrange

        global num_resets
        global env_number
        num_resets += 1
        # Every 5 resets change env and load new spheres positions ffrom /envrionment_gen/
        if num_resets % 5 == 0:
            env_number += 1
            csv_file_path = "environment_generator/generated_envs/{0}/static_obstacles.csv".format(
            "environment_" + str(env_number))
            
            with open(csv_file_path, mode='r') as infile:
                reader = csv.reader(infile)
                # prefab_name,pos_x,pos_y,pos_z,radius
                self.spheres = [[str(rows[0]), float(rows[1]), float(rows[2]), float(rows[3]), float(rows[4])] for rows in
                        reader]
    
        for sphere in self.spheres:
            temp = p.loadURDF(sphere[0],
                            sphere[1:4:],
                            p.getQuaternionFromEuler([0, 0, 0]),
                            physicsClientId=self.CLIENT,
                            useFixedBase=True,
                            globalScaling=10 * sphere[4],
                            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                            )
            p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])


        """
        import pybullet as p
        sphere = p.loadURDF(
            "/home/cam/Desktop/Tutor/SVS/gym-pybullet-drones/experiments/SVS_Code/3D_Models/Hangar/hangar.urdf",
            [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
            useFixedBase=True,
            globalScaling=1 * 0.5,
        )
        """

    def _computeReward(self):
        
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """

        # Implementation with maximum postion reached so far per drone.
        global max_drones_states
        rewards = {}
        drones_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # rewards[1] = -1 * np.linalg.norm(np.array([drones_states[1, 0], drones_states[1, 1], 0.5]) - drones_states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD

        for i in range(0, self.NUM_DRONES):
            min_drone_dist_to_any_sphere = min([np.linalg.norm((np.array([drones_states[i, x] for x in range(3)]) - np.array([s[x] for x in range(1, 4)]))**2) - s[4] for s in self.spheres])
            # Out of thw world margin.
            if drones_states[i, 1] >= WORLD_MARGIN[1][1] or drones_states[i, 1] <= WORLD_MARGIN[1][0] or drones_states[i, 2] >= WORLD_MARGIN[2][1]:
                rewards[i] = -1
            # Collide with any sphere. 
            elif min_drone_dist_to_any_sphere <= 0.1: # Considering a random static drone radius.
                rewards[i] = -1
            # Reached the goal.
            elif drones_states[i, 0] >= GOAL_POSITION[0]:
                rewards[i] = 1
            # Drone actually moved farther then before.
            elif drones_states[i, 0] > max_drones_states[i]:
                rewards[i] = 0.3
                max_drones_states[i] = drones_states[i, 0]
            # Drone moveed back to his maximum position reached.
            elif drones_states[i, 0] < max_drones_states[i]:
                rewards[i] = -0.2
            # Drone still on his maximum position reached so far.
            else:
                rewards[i] = -0.1
        return rewards 


    ################################################################################
    # def _computeReward(self):
    #     WORLD_MARGIN = [[-20,60],[-10,10],[0,10]]
    #     """Computes the current reward value(s).

    #     Returns
    #     -------
    #     dict[int, float]
    #         The reward value for each drone.

    #     """
    #     #using dpp from the class
    #     global dpp
    #     rewards = {}
    #     drones_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     # rewards[1] = -1 * np.linalg.norm(np.array([drones_states[1, 0], drones_states[1, 1], 0.5]) - drones_states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD

    #     for i in range(0, self.NUM_DRONES):
    #         goal_dist = np.linalg.norm(
    #             np.array([drones_states[i, 0], drones_states[i, 1], drones_states[i, 2]]) -  # Drone position.
    #             np.array([GOAL_POSITION[0], drones_states[i, 1], drones_states[i, 2]])  # Goal barrier to surpass.
    #             ) ** 2
    #         #if self.last_drones_dist[i] > sphere_dist and self.last_drones_dist[i] - sphere_dist > 0.2:
    #         #    self.last_drones_dist[i] = sphere_dist
    #         #    rewards[i] = 0.025
    #         #else:
    #         #    rewards[i] = -0.005
    #         if drones_states[i, 1] >= WORLD_MARGIN[1][1] or drones_states[i, 1] <= WORLD_MARGIN[1][0]:
    #             rewards[i] = -1
    #         else:
    #             if drones_states[i, 0] > dpp[i, 0]:
    #                 rewards[i] = 1 - abs(drones_states[i, 1])*0.033 - abs(-5+drones_states[i, 2])*0.066
    #             elif drones_states[i, 0] < dpp[i, 0]:
    #                 rewards[i] = -0.8
    #             else:
    #                 rewards[i] = -0.5
    #     dpp = drones_states[:, 0:3]

    #     return rewards       
             
        
    # def _computeReward(self):
    #     """Computes the current reward value(s).

    #     Returns
    #     -------
    #     dict[int, float]
    #         The reward value for each drone.

    #     """
    #     rewards = {}
    #     states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD

    #     for i in range(0, self.NUM_DRONES):
    #         goal_dist = np.linalg.norm(
    #             np.array([states[i, 0], states[i, 1], states[i, 2]]) -  # Drone position.
    #             np.array([GOAL_POSITION[0], states[i, 1], states[i, 2]])  # Goal barrier to surpass.
    #             ) ** 2
    #         #if self.last_drones_dist[i] > sphere_dist and self.last_drones_dist[i] - sphere_dist > 0.2:
    #         #    self.last_drones_dist[i] = sphere_dist
    #         #    rewards[i] = 0.025
    #         #else:
    #         #    rewards[i] = -0.005
    #         if states[i, 0] > GOAL_POSITION[0]:
    #             rewards[i] = 1
    #         else:
    #             if states[i, 0] < 0:
    #                 rewards[i] = states[i, 0] / 20
    #             else:
    #             #    rewards[i] = 0
    #                 rewards[i] =  states[i, 0] / 60
    #     return rewards

    ################################################################################

    def _computeDone(self):
        ##check the distance between the drone and the goal
        WORLD_MARGIN = [[-20,60],[-10,10],[0,10]]
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        #default > 30
        bool_val = True if self.step_counter / self.SIM_FREQ > 30 else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        if not bool_val:
            drones_pos = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            for i in range(self.NUM_DRONES):
                min_drone_dist_to_any_sphere = min([np.linalg.norm((np.array([drones_pos[i, x] for x in range(3)]) - np.array([s[x] for x in range(1, 4)]))**2) - s[4] for s in self.spheres])
                if drones_pos[i][0] < WORLD_MARGIN[0][0] or drones_pos[i][0] > WORLD_MARGIN[0][1] or drones_pos[i][1] < WORLD_MARGIN[1][0] or drones_pos[i][1] > WORLD_MARGIN[1][1] or drones_pos[i][2] < WORLD_MARGIN[2][0] or drones_pos[i][2] > WORLD_MARGIN[2][1]:
                    done[i] = True
                elif min_drone_dist_to_any_sphere <= 0.1:
                    done[i] = True
                else:
                    done[i] = False
            done["__all__"] = any(done.values())                
        else:
            done["__all__"] = True
            print("Time is up")
        return done              
        # self.last_drones_dist = [1000000 for _ in range(self.NUM_DRONES)]
        # done["__all__"] = bool_val  # True if True in done.values() else False
        # return done
        

    def _observationSpace(self):
        # Latest 10 numbers are for spheres distance from each drone.
        return spaces.Dict({i: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1,0,0,0,0,0,0,0,0,0,0]),
                                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1]),
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})

    def _computeObs(self):                                            
        obs = super()._computeObs() 
        for i in range(self.NUM_DRONES):
            obs[i] = np.concatenate((obs[i], self._computeSphereDist(i)))
        return obs 


    def _computeSphereDist(self, drone):
        drones_pos = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        drone_dist_from_each_spheres = np.array([np.linalg.norm((np.array([drones_pos[drone, x] for x in range(3)]) - np.array([s[x] for x in range(1, 4)]))**2) - s[4] for s in self.spheres])
        drone_dist_from_each_spheres.sort()
        
        drone_dist_from_each_spheres[drone_dist_from_each_spheres > 10] = 10

        return drone_dist_from_each_spheres[:10] / 10

        


    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_X = 3
        MAX_LIN_VEL_Y = 3
        MAX_LIN_VEL_Z = 1

        MAX_X = WORLD_MARGIN[0][1]
        MAX_Y = WORLD_MARGIN[1][1]
        MAX_Z = WORLD_MARGIN[2][1]

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_x = np.clip(state[0], -MAX_X, MAX_X)
        clipped_pos_y = np.clip(state[1], -MAX_Y, MAX_Y)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_x = np.clip(state[10:12], -MAX_LIN_VEL_X, MAX_LIN_VEL_X)
        clipped_vel_y = np.clip(state[10:12], -MAX_LIN_VEL_Y, MAX_LIN_VEL_Y)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_x,
                                               clipped_pos_y,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_x,
                                               clipped_vel_y,
                                               clipped_vel_z
                                               )

        normalized_pos_x = clipped_pos_x / MAX_X
        normalized_pos_y = clipped_pos_y / MAX_Y
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_x = clipped_vel_x / MAX_LIN_VEL_X
        normalized_vel_y = clipped_vel_y / MAX_LIN_VEL_Y
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_x,
                                      normalized_pos_y,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_x,
                                      normalized_vel_y,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(22, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_x,
                                      clipped_pos_y,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_x,
                                      clipped_vel_y,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not (clipped_pos_x == np.array(state[0])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped x position [{:.2f}]".format(state[0]))
        if not (clipped_pos_y == np.array(state[1])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped y position [{:.2f}]".format(state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_x == np.array(state[10])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped x velocity [{:.2f}]".format(state[10]))
        if not (clipped_vel_y == np.array(state[11])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped y velocity [{:.2f}]".format(state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))