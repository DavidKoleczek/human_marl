""" Example usage: hitl_env_basic_plot('plots/hitl_env_basic_data.csv', 'plots/hitl_env_basic_plot.png')
"""
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd


def hitl_env_basic_plot(data_path: str, save_path: str) -> None:
    data = pd.read_csv(data_path).drop('Unnamed: 0', axis=1)
    data_metrics = data[['Penalty', 'Avg Unmodified Returns', 'Avg Interventions']]
    data_std_errs = data[['Penalty', 'Avg Unmodified Returns Std Error', 'Avg Interventions Std Error']]
    data_std_errs = data_std_errs.rename({'Avg Unmodified Returns Std Error': 'Avg Unmodified Returns',
                                          'Avg Interventions Std Error': 'Avg Interventions'}, axis=1)

    data_metrics = data_metrics.melt(id_vars='Penalty', var_name='Metric')
    data_std_errs = data_std_errs.melt(id_vars='Penalty', var_name='Metric')

    data = data_metrics.merge(data_std_errs, how='left', on=['Penalty', 'Metric'])
    data = data.rename({'value_x': 'Value', 'value_y': 'StdError'}, axis=1)

    def errplot(x, y, yerr, **kwargs):
        # source # https://stackoverflow.com/questions/30385975/seaborn-factor-plot-custom-error-bars
        ax = plt.gca()
        to_plot = kwargs.pop("data")
        to_plot.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

    g = sns.FacetGrid(data, col='Metric')
    g.map_dataframe(errplot, 'Penalty', 'Value', 'StdError')

    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle('Gridworld, Vanilla Actor-Critic, LaggyHuman')

    plt.savefig(save_path)
    plt.clf()  # clear matplotlib so we don't get plots on top of each other
