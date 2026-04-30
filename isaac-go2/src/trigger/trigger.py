import logging
import time

import ml_collections
import torch
# import numpy as np

from isaacgym.torch_utils import to_torch
from src.utils.utils import ActionMode, energy_value_2d, check_safety, TriggerType, check_trigger_type
from src.physical_design import MATRIX_P


# torch.set_printoptions(sci_mode=False)


class Trigger:

    def __init__(self, num_envs, trigger_cfg: ml_collections.ConfigDict(), device: str = "cuda"):
        self._device = device
        self._num_envs = num_envs
        self._plant_action = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=device)
        self._action_mode = torch.full((self._num_envs,), ActionMode.STUDENT.value, dtype=torch.int64, device=device)
        self._trigger_type = check_trigger_type(trigger_cfg.trigger_type)
        self._tau = torch.full((self._num_envs,), trigger_cfg.tau, dtype=torch.float32, device=device)
        self._dwell_step = torch.zeros(self._num_envs, dtype=torch.float32, device=device)

        # self._default_epsilon = 1  # Default epsilon
        self._last_action_mode = None
        self._last_action_mode = torch.full((self._num_envs,), ActionMode.STUDENT.value, dtype=torch.int64, device=device)

        # new parameters for dynamic learning space 
        self.eta_max = 0.9 
        self.eta_start = 0.2    
        self.alpha_decay = 0.05 
        
        self._T_n = torch.zeros(self._num_envs, dtype=torch.float32, device=device)
        
        self._eta_k = torch.full((self._num_envs,), self.eta_start, dtype=torch.float32, device=device)
        # =========================================================

    def get_terminal_action(self,
                            stu_action: torch.Tensor,
                            tea_action: torch.Tensor,
                            plant_state: torch.Tensor,
                            learning_space: torch.Tensor,
                            dwell_flag=None):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (phy-teacher/drl-student) to switch for control"""

        if self._trigger_type == TriggerType.SELF:
            terminal_stance_ddq, action_mode = self.self_trig_action(
                stu_action=stu_action,
                tea_action=tea_action,
                plant_state=plant_state,
                learning_space=learning_space,
                dwell_flag=dwell_flag
            )
        elif self._trigger_type == TriggerType.EVENT:
            terminal_stance_ddq, action_mode = self.event_trig_action(
                stu_action=stu_action,
                tea_action=tea_action,
                plant_state=plant_state,
                learning_space=learning_space,
            )
        else:
            raise RuntimeError(f"Unknown trigger type {self._trigger_type}")

        return terminal_stance_ddq, action_mode

    def self_trig_action(self,
                         stu_action: torch.Tensor,
                         tea_action: torch.Tensor,
                         plant_state: torch.Tensor,
                         learning_space: torch.Tensor,
                         dwell_flag=None):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (phy-teacher/drl-student) to switch for control"""

        if dwell_flag is None:
            dwell_flag = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)

        terminal_stance_ddq = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        action_mode = torch.full((self._num_envs,), ActionMode.UNCERTAIN.value, dtype=torch.int64, device=self._device)

        self._last_action_mode = self._action_mode

        # Obtain all energies
        energy_2d = energy_value_2d(plant_state[:, 2:], to_torch(MATRIX_P, device=self._device))

        # apply dynamic boundary
        dynamic_learning_space = learning_space * self._eta_k.unsqueeze(1)

        # check current safety status in the new learning space
        is_unsafe = check_safety(error_state=plant_state, learning_space=dynamic_learning_space)

        # detect precisely when an environment crosses the boundary
        just_triggered = is_unsafe & (self._last_action_mode == ActionMode.STUDENT.value)

        # Increment T_n only for environments that crossed the boundary
        self._T_n = torch.where(just_triggered, self._T_n + 1.0, self._T_n)

        # calculate the new eta_k
        self._eta_k = self.eta_max * (1.0 - (1.0 - self.eta_start) * torch.exp(-self.alpha_decay * self._T_n))
        # =========================================================

        for i, energy in enumerate(energy_2d):

            # Display current system status based on energy
            if is_unsafe[i].item():
                logging.info(f"current system {i} is unsafe")
            else:
                logging.info(f"current system {i} is safe")

            # When Teacher disabled or deactivated
            if not torch.any(tea_action[i]) and bool(dwell_flag[i]) is False:
                logging.info("PHY-Teacher is deactivated, use DRL-Student's action instead")
                self._action_mode[i] = ActionMode.STUDENT.value
                self._plant_action[i] = stu_action[i]

                terminal_stance_ddq[i] = stu_action[i]
                action_mode[i] = ActionMode.STUDENT.value
                continue

            # Teacher activated
            if self._last_action_mode[i] == ActionMode.TEACHER.value:

                # Teacher Dwell time
                if dwell_flag[i]:
                    if tea_action[i] is None:
                        raise RuntimeError(f"Unrecognized PHY-Teacher action {tea_action[i]} from {i} for dwelling")
                    else:
                        logging.info("Continue PHY-Teacher action in dwell time")
                        self._action_mode[i] = ActionMode.TEACHER.value
                        self._plant_action[i] = tea_action[i]

                        terminal_stance_ddq[i] = tea_action[i]
                        action_mode[i] = ActionMode.TEACHER.value

                # Switch back to HPC
                else:
                    self._action_mode[i] = ActionMode.STUDENT.value
                    self._plant_action[i] = stu_action[i]
                    logging.info(f"Max PHY-Teacher dwell time achieved, switch back to DRL-Student control")

                    terminal_stance_ddq[i] = stu_action[i]
                    action_mode[i] = ActionMode.STUDENT.value

            elif self._last_action_mode[i] == ActionMode.STUDENT.value:

                # Inside safety subset
                if not is_unsafe[i].item():
                    self._action_mode[i] = ActionMode.STUDENT.value
                    self._plant_action[i] = stu_action[i]
                    logging.info(f"Continue DRL-Student action")

                    terminal_stance_ddq[i] = stu_action[i]
                    action_mode[i] = ActionMode.STUDENT.value

                # Outside safety envelope (bounded by epsilon)
                else:
                    logging.info(f"Switch to PHY-Teacher action for safety concern")
                    self._action_mode[i] = ActionMode.TEACHER.value
                    self._plant_action[i] = tea_action[i]

                    terminal_stance_ddq[i] = tea_action[i]
                    action_mode[i] = ActionMode.TEACHER.value
            else:
                raise RuntimeError(f"Unrecognized last action mode: {self._last_action_mode[i]} for {i}")

        return terminal_stance_ddq, action_mode

    def event_trig_action(self,
                          stu_action: torch.Tensor,
                          tea_action: torch.Tensor,
                          plant_state: torch.Tensor,
                          learning_space: torch.Tensor):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (phy-teacher/drl-student) to switch for control"""

        terminal_stance_ddq = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        action_mode = torch.full((self._num_envs,), ActionMode.UNCERTAIN.value, dtype=torch.int64, device=self._device)

        self._last_action_mode = self._action_mode

        # Obtain all energies
        energy_2d = energy_value_2d(plant_state[:, 2:], to_torch(MATRIX_P, device=self._device))

        # apply dynamic boundry
        dynamic_learning_space = learning_space * self._eta_k.unsqueeze(1)

        # check safety against the new learning space
        is_unsafe = check_safety(error_state=plant_state, learning_space=dynamic_learning_space)

        # check when an environment crosses the boundary
        just_triggered = is_unsafe & (self._last_action_mode == ActionMode.STUDENT.value)

        # Increment T_n only for environments that just crossed the boundary
        self._T_n = torch.where(just_triggered, self._T_n + 1.0, self._T_n)

        # Recalculate eta_k for all environments simultaneously based on their individual T_n
        self._eta_k = self.eta_max * (1.0 - (1.0 - self.eta_start) * torch.exp(-self.alpha_decay * self._T_n))
        # =========================================================

        teacher_deactivated = ~torch.any(tea_action != 0, dim=1)

   
        use_student = teacher_deactivated | ~is_unsafe

        # Assign action modes instantly across all envs using boolean masks
        self._action_mode = torch.where(use_student, 
                                        ActionMode.STUDENT.value, 
                                        ActionMode.TEACHER.value)

        terminal_stance_ddq = torch.where(use_student.unsqueeze(1), stu_action, tea_action)
        
        # Update internal state tracking
        self._plant_action = terminal_stance_ddq.clone()

        return terminal_stance_ddq, self._action_mode
    
    def reset_idx(self, env_ids: torch.Tensor):
        """
        Resets the dynamic boundary tracking and action modes 
        for specific environments that have terminated.
        """
        if len(env_ids) == 0:
            return

        # 1. Reset the trigger counter back to 0
        self._T_n[env_ids] = 0.0

        # 2. Reset the dynamic boundary back to its constricted starting size
        self._eta_k[env_ids] = self.eta_start
        
        # 3. Reset the dwell step counter (from your existing code)
        self._dwell_step[env_ids] = 0.0
        
        # 4. Safely return control to the student for the new episode
        self._action_mode[env_ids] = ActionMode.STUDENT.value
        self._last_action_mode[env_ids] = ActionMode.STUDENT.value
        self._plant_action[env_ids] = 0.0

    @property
    def device(self):
        return self._device

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode
