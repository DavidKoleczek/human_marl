from utils.single_env_experiment import SingleEnvExperiment
from all.experiments.parallel_env_experiment import ParallelEnvExperiment


def run_experiment(
        agents,
        envs,
        frames,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        write_loss=True,
        max_steps = 200
):
    if not isinstance(agents, list):
        agents = [agents]

    if not isinstance(envs, list):
        envs = [envs]

    for env in envs:
        for agent in agents:
            make_experiment = get_experiment_type(agent)
            experiment = make_experiment(
                agent,
                env,
                logdir=logdir,
                quiet=quiet,
                render=render,
                write_loss=write_loss,
                max_steps=max_steps
            )
            experiment.train(frames=frames)
            experiment.test(episodes=test_episodes)


def get_experiment_type(agent):
    if is_parallel_env_agent(agent):
        return ParallelEnvExperiment
    return SingleEnvExperiment


def is_parallel_env_agent(agent):
    return isinstance(agent, tuple)
