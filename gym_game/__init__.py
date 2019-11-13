from gym.envs.registration import register

register(
    id='Pygame-v0',
    entry_point='gym_game.envs:GameEnv',
    max_episode_steps=9999,
)
