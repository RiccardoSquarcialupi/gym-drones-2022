import collections
import logging
import sys

import cv2
import numpy as np
from gym import spaces

sys.path.append('../')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.utils.utils import nnlsRPM
import pybullet as p

# x y z of each drone, concatenated with 2 booleans indicating whether the drone can see an enemy and if he is shooting.
BLUE_ARRAY_TARGET = np.array([[0, 0, 0, False, False],[0, 0, 0, False, False],[0, 0, 0, False, False],[0, 0, 0, False, False]])
RED_ARRAY_TARGET = np.array([[0, 0, 0, False, False],[0, 0, 0, False, False],[0, 0, 0, False, False],[0, 0, 0, False, False]])

# resolution of virtual camera of each drone.
CAMERA_VISION = [160, 80]

# number of spheres in each drone that can be shooted.
NUM_SPHERES = np.array([5,5,5,5,5,5,5,5])

class BattleAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL):
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
        self.last_drones_dist = [1000000 for _ in range(self.NUM_DRONES)]
        self.IMG_RES = np.array(CAMERA_VISION)
        # maximum duration of a single episode in step.
        self.max_steps = 240 * 30
        # array of flying spheres for each drone, stored to be deleted when touching the ground.
        self.drones_sphere = [np.array([], dtype=np.int32) for _ in range(self.NUM_DRONES)]
        if not p.isNumpyEnabled():
            logging.warning("Numpy speed-up camera, try to activate it!")
            p.enableNumpySpeedup()

    ################################################################################

    def reset(self):
        return super().reset()

    def _addObstacles(self):

        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """

        import os
        import pybullet as p
        global NUM_SPHERES
        # disable shadows for better images
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        # add hangar urdf
        self.drones_spheres = [np.array([], dtype=np.int32) for _ in range(self.NUM_DRONES)]
        # reset number of spheres in each drone.
        for i in range(self.NUM_DRONES):
            NUM_SPHERES[i] = 5
        # assigning each drone to a team.
        for i in range(1, self.NUM_DRONES + 1):
            if i > self.NUM_DRONES / 2:
                # red team
                color = [1, 0, 0, 1]
            else:
                # blue team
                color = [0, 0, 1, 1]
            p.changeVisualShape(i, -1, rgbaColor=color, physicsClientId=self.CLIENT)

        # disambiguation: in this way it works both called from exercise-2 folder and father folder.
        if os.path.exists('exercise-2/Hangar/hangar.urdf'):
            p.loadURDF(
                "exercise-2/Hangar/hangar.urdf",
                [-0, -0, 1],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                globalScaling=0.05,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
        else:
            p.loadURDF(
                "Hangar/hangar.urdf",
                [-0, -0, 1],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                globalScaling=0.05,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )

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

    ################################################################################
    def _saveLastAction(self,
                        action
                        ):
        """Stores the most recent action into attribute `self.last_action`.

        The last action can be used to compute aerodynamic effects.
        The method disambiguates between array and dict inputs
        (for single or multi-agent aviaries, respectively).

        Parameters
        ----------
        action : ndarray | dict
            (4)-shaped array of ints (or dictionary of arrays) containing the current RPMs input.

        """
        if isinstance(action, collections.abc.Mapping):
            for k, v in action.items():
                new_v = v["guide_space"]
                res_v = np.resize(new_v, (
                    1, 4))  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses

                self.last_action[int(k), :] = res_v
        else:
            res_action = np.resize(action, (
                1, 4))  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
            self.last_action = np.reshape(res_action, (len(self._agent_ids), 4))

    ################################################################################
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict() of Box() of size 1, 3, or 3, depending on the action type,
            indexed by drone Id in integer format.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BattleAviary._actionSpace()")
            exit()

        # used to add the new action shoot the sphere
        return spaces.Dict({i: spaces.Dict({"guide_space": spaces.Box(low=-1 * np.ones(size),
                                                                      high=np.ones(size),
                                                                      dtype=np.float32
                                                                      ), "shoot_space": spaces.discrete.Discrete(2)})
                            for i in range(self.NUM_DRONES)})

    ################################################################################
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : dict[str, ndarray]
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # The action of the dead drones are set to 0
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k, value in action.items():
            # using old code with the chooose guide action
            v = value["guide_space"]
            if self.ACT_TYPE == ActionType.RPM:

                rpm[int(k), :] = np.array(self.HOVER_RPM * (1 + 0.05 * v))
            elif self.ACT_TYPE == ActionType.DYN:
                rpm[int(k), :] = nnlsRPM(thrust=(self.GRAVITY * (v[0] + 1)),
                                         x_torque=(0.05 * self.MAX_XY_TORQUE * v[1]),
                                         y_torque=(0.05 * self.MAX_XY_TORQUE * v[2]),
                                         z_torque=(0.05 * self.MAX_Z_TORQUE * v[3]),
                                         counter=self.step_counter,
                                         max_thrust=self.MAX_THRUST,
                                         max_xy_torque=self.MAX_XY_TORQUE,
                                         max_z_torque=self.MAX_Z_TORQUE,
                                         a=self.A,
                                         inv_a=self.INV_A,
                                         b_coeff=self.B_COEFF,
                                         gui=self.GUI
                                         )
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(int(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                                                               cur_pos=state[0:3],
                                                               cur_quat=state[3:7],
                                                               cur_vel=state[10:13],
                                                               cur_ang_vel=state[13:16],
                                                               target_pos=state[0:3] + 0.1 * v
                                                               )
                rpm[int(k), :] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(int(k))
                if np.linalg.norm(v[0:3]) != 0:
                    v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                                                              cur_pos=state[0:3],
                                                              cur_quat=state[3:7],
                                                              cur_vel=state[10:13],
                                                              cur_ang_vel=state[13:16],
                                                              target_pos=state[0:3],  # same as the current position
                                                              target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                              target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector
                                                              # target the desired velocity vector
                                                              )
                rpm[int(k), :] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[int(k), :] = np.repeat(self.HOVER_RPM * (1 + 0.05 * v), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_DYN:
                rpm[int(k), :] = nnlsRPM(thrust=(self.GRAVITY * (1 + 0.05 * v[0])),
                                         x_torque=0,
                                         y_torque=0,
                                         z_torque=0,
                                         counter=self.step_counter,
                                         max_thrust=self.MAX_THRUST,
                                         max_xy_torque=self.MAX_XY_TORQUE,
                                         max_z_torque=self.MAX_Z_TORQUE,
                                         a=self.A,
                                         inv_a=self.INV_A,
                                         b_coeff=self.B_COEFF,
                                         gui=self.GUI
                                         )
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(int(k))
                rpm, _, _ = self.ctrl[k].computeControl(control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3] + 0.1 * np.array([0, 0, v[0]])
                                                        )
                rpm[int(k), :] = rpm
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool = True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(fov=45.0,
                                                     aspect=1.0,
                                                     nearVal=self.L,
                                                     farVal=1000.0
                                                     )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, _, _] = p.getCameraImage(width=self.IMG_RES[0],
                                             height=self.IMG_RES[1],
                                             shadow=0,
                                             viewMatrix=DRONE_CAM_VIEW,
                                             projectionMatrix=DRONE_CAM_PRO,
                                             flags=SEG_FLAG,
                                             physicsClientId=self.CLIENT
                                             )
        rgb = np.reshape(rgb, (h, w, 4))

        return rgb

    # compute image for each drone, setting target array.
    def setupForStep(self,drones):
        if drones in [4,5,6,7]:
            img_red = self._getDroneImages(drones, False)
            bgr_red = cv2.cvtColor(img_red, cv2.COLOR_BGRA2BGR)
            hsv_red = cv2.cvtColor(bgr_red, cv2.COLOR_BGR2HSV)
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            mask_red = cv2.inRange(hsv_red, red_lower, red_upper)
            result_red = cv2.bitwise_and(img_red, img_red, mask=mask_red)
            mask = mask_red

            if mask.any() != 0:
                for i in range(0, CAMERA_VISION[1]):
                    for j in range(0, CAMERA_VISION[0]):
                        if mask[i, j] != 0:
                            RED_ARRAY_TARGET[drones%4, 0:3] = np.array([100, j - (CAMERA_VISION[0] / 2), i - (CAMERA_VISION[1] / 2)])
                            break
                    if RED_ARRAY_TARGET[drones%4, 0] == 100:
                        RED_ARRAY_TARGET[drones%4, 3] = True
                        break

        else:
            img_blue = self._getDroneImages(drones, False)
            bgr_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGRA2BGR)
            hsv_blue = cv2.cvtColor(bgr_blue, cv2.COLOR_BGR2HSV)
            blue_upper = np.array([180, 255, 255])
            blue_lower = np.array([60, 35, 140])
            mask_blue = cv2.inRange(hsv_blue, blue_lower, blue_upper)
            mask = mask_blue
            
            if mask.any() != 0:
                for i in range(0, CAMERA_VISION[1]):
                    for j in range(0, CAMERA_VISION[0]):
                        if mask[i, j] != 0:
                            BLUE_ARRAY_TARGET[drones, 0:3] = np.array([100, j - (CAMERA_VISION[0] / 2), i - (CAMERA_VISION[1] / 2)])
                            break
                    if BLUE_ARRAY_TARGET[drones, 0] == 100:
                        BLUE_ARRAY_TARGET[drones, 3] = True
                        break

            

        

    def step(self, action):
        global RED_ARRAY_TARGET
        global BLUE_ARRAY_TARGET
        global NUM_SPHERES

        RED_ARRAY_TARGET[:,3] = False
        BLUE_ARRAY_TARGET[:,3] = False

        
        # Visualize the image
        for i in range(0, self.NUM_DRONES):
            self.setupForStep(i)
        # cv2.imshow('image_drone_red', result_red)
        # cv2.imshow('mask_drone_red', mask_red)

        
        #result_blue = cv2.bitwise_and(img_blue, img_blue, mask=mask_blue)
        # cv2.imshow('image_drone_blue', result_blue)
        # cv2.imshow('mask_drone_blue', mask_blue)

        #cv2.waitKey(10)
        # print("IMG: " + str(img.shape))
        # print("FILTER: " + str(filter.shape))
        # print(filter)
        # print(filter)
        #import matplotlib.pyplot as plt
        # plt.imshow(img_red, cmap='gray', vmin=0, vmax=255)
        # plt.pause(0.0001)
        # plt.show(block=False)
        # plt.imshow(img_blue, cmap='gray', vmin=0, vmax=255)
        # plt.pause(0.0001)
        # plt.show(block=False)

        # check if the red drone is in the image
        # print(action)
        # check if the blue drone is in the image
        

        RED_ARRAY_TARGET[:,4] = False
        BLUE_ARRAY_TARGET[:,4] = False

        for drone_index, gym_dict in action.items():
            # if drones want to shoot
            if gym_dict["shoot_space"] == 1:
                # if drone is red team and can shoot.
                if RED_ARRAY_TARGET[drone_index%4,3] and drone_index in [4,5,6,7] and NUM_SPHERES[drone_index] > 0:
                    RED_ARRAY_TARGET[drone_index%4,4] = True
                    self.drones_spheres[drone_index] = np.append(self.drones_spheres[drone_index],
                                                                 self._drone_shoot(drone_index))
                # if drone is blue team and can shoot.
                if BLUE_ARRAY_TARGET[drone_index%4,3] and drone_index in [0,1,2,3] and NUM_SPHERES[drone_index] > 0:
                    BLUE_ARRAY_TARGET[drone_index%4,4] = True
                    self.drones_spheres[drone_index] = np.append(self.drones_spheres[drone_index],
                                                                 self._drone_shoot(drone_index))

            if gym_dict["shoot_space"] == 1 and NUM_SPHERES[drone_index] == 0:
                # Spheres to shoot are over, set it to -1, so we can negative reward the drone for shooting.
                NUM_SPHERES[drone_index] = -1

            # each flying sphere currently touching the ground is removed
            # nice performance improvement.
            try:
                removable_spheres = np.array([], dtype=np.int32)
                for sphere in self.drones_spheres[drone_index]:
                    if (p.getBasePositionAndOrientation(sphere)[0][2] < 1.1):
                        p.removeBody(sphere)
                        removable_spheres = np.append(removable_spheres, sphere)
                self.drones_spheres[drone_index] = np.setdiff1d(self.drones_spheres[drone_index], removable_spheres)
            except:
                pass

        # reset want to shoot
        RED_ARRAY_TARGET[:,0:3] = 0
        BLUE_ARRAY_TARGET[:,0:3] = 0
        return super().step(action)

    # convert a 2D point in camera to world 3D coordinates.
    def from2Dto3D(self, array_target):
        f_x = 1000.
        f_y = 1000.
        c_x = 1000.
        c_y = 1000.
        Z = array_target[0]
        Z = [Z]
        x = array_target[1]
        y = array_target[2]
        intrinsic = np.array([
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0]
        ])

        distortion = np.array([0.0, 0.0, 0.0, 0.0])

        f_x = intrinsic[0, 0]
        f_y = intrinsic[1, 1]
        c_x = intrinsic[0, 2]
        c_y = intrinsic[1, 2]
        # This was an error before
        # c_x = intrinsic[0, 3]
        # c_y = intrinsic[1, 3]

        # Step 1. Undistort.
        points_undistorted = np.array([])

        points_undistorted = cv2.undistortPoints(np.expand_dims(np.array([x, y]).astype(np.float32), axis=1), intrinsic,
                                                 distortion, P=intrinsic)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

        # Step 2. Reproject.
        result = []
        for idx in range(points_undistorted.shape[0]):
            z = Z[0] if len(Z) == 1 else Z[idx]
            x = (points_undistorted[idx, 0] - c_x) / f_x * z
            y = (points_undistorted[idx, 1] - c_y) / f_y * z
            result.append([x, y, z])
        return result[0]

    def _drone_shoot(self, drone_index):
        import pybullet as p
        global RED_ARRAY_TARGET
        global BLUE_ARRAY_TARGET 
        global NUM_SPHERES

        NUM_SPHERES[drone_index] -= 1

        shoot_target = (
            self.from2Dto3D(BLUE_ARRAY_TARGET[drone_index,0:3]) 
            if drone_index in [0,1,2,3] else
            self.from2Dto3D(RED_ARRAY_TARGET[drone_index%4,0:3]))
        # print("shoot_target", shoot_target)
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[drone_index, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, shoot_target) + np.array(self.pos[drone_index, :])
        # fix drone index, 0 is the floor
        drone_index += 1
        # print(target)
        pos_and_orientation = p.getBasePositionAndOrientation(drone_index)
        # print("#############################################################################")
        # print("pos_and_orientation", pos_and_orientation)
        # print("#############################################################################")
        pos = np.array(pos_and_orientation[0]) + 0.2
        temp = p.loadURDF("sphere_small.urdf",
                          pos,
                          p.getQuaternionFromEuler([0, 0, 0]),
                          physicsClientId=self.CLIENT,
                          useFixedBase=False,
                          globalScaling=2,
                          flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                          )
        # print("Sphere velocity = " + str(p.getBaseVelocity(drone_index, physicsClientId=self.CLIENT)))
        # projectivle are black
        p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.resetBaseVelocity(temp, target, [0, 0, 0], physicsClientId=self.CLIENT)
        return temp

    ################################################################################
    def _computeReward(self):
        global BLUE_ARRAY_TARGET
        global RED_ARRAY_TARGET
        global NUM_SPHERES

        array_target = np.concatenate((BLUE_ARRAY_TARGET, RED_ARRAY_TARGET), axis=0)
        # print("############################")
        # print(BLUE_ARRAY_TARGET)
        # print(RED_ARRAY_TARGET)
        # print(array_target)
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD

        for i in range(self.NUM_DRONES):
            if states[i, 2] < 0.5:  # sono stato atterrato
                rewards[i] = -1
                #if self.step_counter/self.SIM_FREQ < 3
            elif NUM_SPHERES[i] == -1: # Shooting a sphere when they are ended.
                NUM_SPHERES[i] = 0
                rewards[i] = -0.5
            elif self.step_counter <= 240*3 and array_target[i][4]: # sparo nei primi 3 secondi.
                rewards[i] = -1
            elif array_target[i][3] and array_target[i][4]:  # vedo e sparo
                if states[(i + 1) % 2, 2] < 0.5:  # ho atterrato
                    rewards[i] = 1  # ho ucciso
                else:
                    rewards[i] = 0.1  # ho visto e sparato
            elif array_target[i][3] and array_target[i][4] == False:  # vedo e ma non sparo
                rewards[i] = 0.05  # ho visto
            elif array_target[i][3] == False and array_target[i][4]:  # non vedo e sparo
                rewards[i] = -0.1
            else:  # non vedo e non sparo
                rewards[i] = -0.05
        return rewards

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        done = {i: self.step_counter>self.max_steps for i in range(self.NUM_DRONES)}
        done["__all__"] = False

        #if self.step_counter == 300:
        #    done[0] = True
        #    self._agent_ids.remove(0)
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if states[i, 2] < 0.3:  # if drone is on the ground
                done[1] = True
                done["__all__"] = True
        if self.step_counter > self.max_steps:
            done[1] = True
            done["__all__"] = True
        return done

    ################################################################################

    def _observationSpace(self):
        low_0 = [0] * CAMERA_VISION[0] * CAMERA_VISION[1] * 4
        low = np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1] + low_0 + [-1] + [0])
        high_1 = [1] * CAMERA_VISION[0] * CAMERA_VISION[1] * 4
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + high_1 + [5] + [1])
        return spaces.Dict({i: spaces.Box(low=low,
                                          high=high,
                                          dtype=np.float64
                                          ) for i in range(self.NUM_DRONES)})

    def _computeObs(self):
        global NUM_SPHERES
        obs = super()._computeObs()
        for i in obs.keys():
            # print(self._getDroneImages(i, False).flatten)
            obs[i] = np.concatenate(
                (
                    obs[i],
                    np.array(self._getDroneImages(i, False) / 255, dtype=np.float64).flat, # concatenate image matrix
                    [NUM_SPHERES[i]], # and spheres num
                    [self.step_counter / self.max_steps] # and step counter for time passed from the beginning.
                )
            ) 
                                                      
        return obs

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
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                      state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                      state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
