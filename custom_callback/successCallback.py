from stable_baselines3.common.callbacks import BaseCallback

class SuccessCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(SuccessCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.success_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Assume 'done' is a boolean that is True when an episode is successful
            done = self.locals.get('done')
            if done:
                self.success_count += 1
            else:
                self.success_count = 0  # Reset count if not successful

            if self.success_count >= 100:
                return False  # Return False to stop training

        return True  # Return True to continue training