from gym.envs.registration import register

register(
    id='Game-v0',
    entry_point='gym_game.envs:Env',
    max_episode_steps=9999,
)
