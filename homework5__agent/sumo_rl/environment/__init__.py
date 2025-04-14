"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register


register(
    id="sumo-rl1-v0",
    entry_point="sumo_rl.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
