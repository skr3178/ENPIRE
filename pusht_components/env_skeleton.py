# Illustrative env.py skeleton shown on the ENPIRE website (research.nvidia.com/labs/gear/enpire)
# This is the abstraction a coding agent fills in: reset, reward, observation, step.
# (Pseudocode as displayed on the site — NOT the real gym-pusht implementation.)

class InsertionEnv:
    def reset(self):
        # TODO: auto task reset
        pick_and_place(obj, target)
        go_home()
        ...

    def get_reward(self, obs, act):
        # TODO: scalar reward
        mask = sam3(obs['left'])
        pos = bound_sdf(obs, mask)
        ...

    def get_observation(self):
        ...

    def step(self, act):
        ...
