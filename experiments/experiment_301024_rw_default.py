import optuna
import pandas as pd
import mesa


# Import RandomWalker-specific models and strategies
from abm.model import Model as RandomWalkerModel
from abm.exploration_strategy import RandomWalkerExplorationStrategy
from abm.exploitation_strategy import ExploitationStrategy


def objective(trial):
    """
    The objective function that Optuna will optimize.
    It defines the parameters for both GP and Random Walker models
    and computes the average collected resource based on the exploration and exploitation strategies.
    """
    
    grid_size = 20
    # Define the hyperparameters for the exploration strategy based on the model type
    L = grid_size   # Maximum distance for Levy flight
    d_min = 10^-3   # Minimum distance for Levy flight
        
    #dmin = trial.suggest_float("dmin", 0.1, 5.0)  # Minimum distance for Levy flight
    #L = trial.suggest_float("L", 5.0, 50.0)  # Maximum distance for Levy flight
    
    # Actual hyperparameters 

    mu = trial.suggest_float("mu", 1.1, 3.0)  # Exponent for Levy flight
    alpha = trial.suggest_float("alpha", 0.01, 1.0)  # Parameter for social cue adjustment
    threshold = trial.suggest_int("threshold", 1, 20)

    print('Model type: Random Walker')

    # Set up Random Walker exploration strategy
    exploration_strategy = RandomWalkerExplorationStrategy(
        mu=mu,
        dmin=dmin,
        L=L,
        alpha=alpha,
        grid_size=grid_size
    )
    exploitation_strategy = PatchEvaluationSubroutine(threshold=threshold)
    model = RandomWalkerModel

    # Run the simulation using Mesa's batch_run
    results = mesa.batch_run(
        model,
        parameters={
            "exploration_strategy": exploration_strategy,
            "exploitation_strategy": exploitation_strategy,
            "grid_size": grid_size,
            "number_of_agents": 5,
            "n_resource_clusters": 2,
            "resource_quality": 0.8,
            "resource_cluster_radius": 2,
            "keep_overall_abundance": True,
        },
        iterations=100,
        number_processes=None,  # use all CPUs
        max_steps=100,
        data_collection_period=-1,  # only the last step
    )
    results = pd.DataFrame(results)

    # Filter out agents (resource AgentID is usually 0, so we remove it)
    mask = results.AgentID != 0

    # Calculate the average collected resource
    avg_collected_resource = results.loc[mask, 'collected_resource'].mean()

    return avg_collected_resource


if __name__ == '__main__':
    # Create the Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, n_jobs=-1)

    # Print the best trial results
    trial = study.best_trial
    print("Average collected resource: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    # Visualization: Plot the optimization history
    import plotly.io as pio
    pio.templates['plotly'].layout['autosize'] = False

    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(height=400, width=1200)
    fig.show()

    # Slice plot (visualize the effects of individual parameters)
    params = ['sigma', 'mu', 'dmin', 'L', 'C', 'alpha', 'threshold']

    fig = optuna.visualization.plot_slice(study, params=params)
    fig.show()

    # Contour plot (visualize parameter interactions)
    fig = optuna.visualization.plot_contour(study, params=params)
    fig.update_layout(height=800, width=1200)
    fig.show()

    # Parameter importances (visualize which parameters contribute most to the objective)
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(height=400, width=1200)
    fig.show()
