import numpy as np
import pytest
from pytest import fixture
from ice_fishing_abm_1.ice_fishing_abm_random_walker.movement_destination_subroutine import ExplorationStrategy, \
    RandomWalkerExplorationStrategy

@fixture
def random_walker_strategy():
    return RandomWalkerExplorationStrategy(grid_size=100, motion_type='Brownian', sigma=1.0)

@pytest.mark.parametrize("motion_type", ['Brownian', 'Levy'])
def test_choose_destination_random_walker(motion_type):
    """
    Test that the destination selected by the random walker is within grid bounds and checks for both Brownian and Levy motion types.
    """
    walker_model = RandomWalkerExplorationStrategy(motion_type=motion_type, grid_size=100)
    
    # Define empty input cases for simplicity
    social_locs = np.empty((0, 2))
    catch_locs = np.empty((0, 2))
    loss_locs = np.empty((0, 2))
    
    # Ensure destination is calculated and within bounds for both motion types
    destination = walker_model.choose_destination(social_locs, catch_locs, loss_locs)
    assert destination.shape == (2,)
    assert 0 <= destination[0] < 100
    assert 0 <= destination[1] < 100


@pytest.mark.parametrize("sigma", [0.5, 1.0, 5.0])
def test_brownian_motion_distance(sigma):
    """
    Test that the distance sampled for Brownian motion is within expected limits, using different values of sigma.
    """
    walker_model = RandomWalkerExplorationStrategy(motion_type='Brownian', sigma=sigma, grid_size=100)
    
    # Run Brownian motion sampling
    destination = walker_model._brownian_motion()
    
    # The destination should be valid and within the grid bounds
    assert destination.shape == (2,)
    assert 0 <= destination[0] < 100
    assert 0 <= destination[1] < 100
    
    # Manually compute expected step size range and compare
    distance_travelled = np.linalg.norm(np.array([50, 50]) - destination)
    assert distance_travelled <= 4 * sigma  # 4*sigma is an arbitrary upper bound for reasonable displacement


@pytest.mark.parametrize("mu, C, dmin, L", [(1.5, 1.0, 1.0, 10.0), (2.0, 2.0, 1.0, 20.0)])
def test_levy_flight_distance(mu, C, dmin, L):
    """
    Test that the distance sampled for Levy flight is within expected limits for different values of mu, C, dmin, and L.
    """
    walker_model = RandomWalkerExplorationStrategy(motion_type='Levy', mu=mu, C=C, dmin=dmin, L=L, grid_size=100)
    
    # Run Levy flight sampling
    destination = walker_model._levy_flight()
    
    # The destination should be valid and within the grid bounds
    assert destination.shape == (2,)
    assert 0 <= destination[0] < 100
    assert 0 <= destination[1] < 100
    
    # Manually compute the range of expected distance and compare
    distance_travelled = np.linalg.norm(np.array([50, 50]) - destination)
    assert dmin <= distance_travelled <= L  # Ensure that the distance lies within the expected range


def test_social_cue_adjustment():
    """
    Test the random walker adjustment when social cues are detected. 
    Ensure the walker is more likely to move toward social cues if they are nearby.
    """
    walker_model = RandomWalkerExplorationStrategy(grid_size=100, motion_type='Brownian', sigma=1.0)
    
    # Define the current destination and social locations
    current_destination = np.array([60, 60])
    social_locs = np.array([[58, 58], [40, 40], [70, 70]])  # One close social location, others far
    
    # Adjust for social cue
    adjusted_destination = walker_model._adjust_for_social_cue(current_destination, social_locs)
    
    # Since there's a social cue near (58, 58), check if the destination was adjusted
    delta_d = np.linalg.norm(adjusted_destination - current_destination)
    
    # We expect the walker to move closer to the social cue, so the new delta should be less than the original
    original_delta_d = np.linalg.norm(current_destination - social_locs, axis=1).min()
    assert delta_d <= original_delta_d


def test_brownian_motion_zero_step():
    """
    Test that the Brownian motion step size is non-zero when sigma is set to a positive value.
    """
    walker_model = RandomWalkerExplorationStrategy(motion_type='Brownian', sigma=0.5, grid_size=100)
    
    # Sample a destination
    destination = walker_model._brownian_motion()
    
    # The destination should not be the same as the starting point (50, 50)
    assert not np.allclose(destination, np.array([50, 50]))


def test_levy_flight_zero_step():
    """
    Test that Levy flight generates valid steps and avoids zero distance.
    """
    walker_model = RandomWalkerExplorationStrategy(motion_type='Levy', mu=1.5, C=1.0, dmin=1.0, L=10.0, grid_size=100)
    
    # Sample a destination
    destination = walker_model._levy_flight()
    
    # The distance travelled should not be zero
    assert not np.allclose(destination, np.array([50, 50]))


@pytest.mark.parametrize("motion_type", ['Brownian', 'Levy'])
def test_edge_case_small_grid(motion_type):
    """
    Test for edge cases when using a small grid size (e.g., 2x2) for Brownian and Levy motion.
    """
    walker_model = RandomWalkerExplorationStrategy(motion_type=motion_type, grid_size=2)
    
    # Define empty inputs for simplicity
    social_locs = np.empty((0, 2))
    catch_locs = np.empty((0, 2))
    loss_locs = np.empty((0, 2))
    
    # Ensure destination is calculated and within small grid bounds
    destination = walker_model.choose_destination(social_locs, catch_locs, loss_locs)
    assert destination.shape == (2,)
    assert 0 <= destination[0] < 2
    assert 0 <= destination[1] < 2
