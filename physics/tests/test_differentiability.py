"""
Tests for Differentiability

This module verifies that the physics simulation is fully differentiable
with respect to all relevant parameters, which is essential for:
1. Training neural networks end-to-end through the physics
2. Gradient-based optimization of force field parameters
3. Sensitivity analysis of trajectories

Tests use JAX's grad and jacfwd/jacrev to verify gradient computation.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
import numpy as np

# Import physics modules
from physics.state import PhysicsState, SimulationConfig, create_initial_state
from physics.fields import (
    WindField,
    VortexField,
    PointForce,
    GravityWell,
    DragZone,
    soft_indicator_box,
    soft_indicator_circle,
)
from physics.simulator import (
    simulate_trajectory,
    simulate_positions_only,
    compute_total_force,
    trajectory_loss_same_length,
    semi_implicit_euler_step,
)
from physics.utils import (
    generate_line_trajectory,
    generate_arc_trajectory,
    compute_trajectory_length,
)


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture
def simple_config():
    """Simple configuration for fast tests."""
    return SimulationConfig(
        dt=0.01,
        num_steps=50,
        ball_mass=1.0,
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        boundary_softness=0.2,
        boundary_strength=10.0,
    )


@pytest.fixture
def initial_state():
    """Default initial state."""
    return create_initial_state(x=0.0, y=0.0, vx=1.0, vy=0.0)


# =============================================================================
# Soft Indicator Function Tests
# =============================================================================

class TestSoftIndicators:
    """Test soft indicator functions for smoothness and differentiability."""
    
    def test_soft_indicator_box_inside(self):
        """Test that indicator is ~1 inside the box."""
        center = jnp.array([0.0, 0.0])
        half_size = jnp.array([1.0, 1.0])
        position = jnp.array([0.0, 0.0])
        
        value = soft_indicator_box(position, center, half_size, softness=0.1)
        assert value > 0.9, f"Expected ~1 inside box, got {value}"
    
    def test_soft_indicator_box_outside(self):
        """Test that indicator is ~0 outside the box."""
        center = jnp.array([0.0, 0.0])
        half_size = jnp.array([1.0, 1.0])
        position = jnp.array([3.0, 3.0])
        
        value = soft_indicator_box(position, center, half_size, softness=0.1)
        assert value < 0.1, f"Expected ~0 outside box, got {value}"
    
    def test_soft_indicator_box_differentiable(self):
        """Test that box indicator has smooth gradients."""
        center = jnp.array([0.0, 0.0])
        half_size = jnp.array([1.0, 1.0])
        
        def indicator_fn(pos):
            return soft_indicator_box(pos, center, half_size, softness=0.1)
        
        # Compute gradient at boundary
        position = jnp.array([0.9, 0.0])
        grad_fn = grad(indicator_fn)
        gradient = grad_fn(position)
        
        # Gradient should exist and be finite
        assert jnp.all(jnp.isfinite(gradient)), "Gradient should be finite"
        # Gradient should point inward (negative x direction)
        assert gradient[0] < 0, "Gradient should point inward"
    
    def test_soft_indicator_circle_differentiable(self):
        """Test that circle indicator has smooth gradients."""
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        
        def indicator_fn(pos):
            return soft_indicator_circle(pos, center, radius, softness=0.1)
        
        # Compute gradient at boundary
        position = jnp.array([0.9, 0.0])
        grad_fn = grad(indicator_fn)
        gradient = grad_fn(position)
        
        assert jnp.all(jnp.isfinite(gradient)), "Gradient should be finite"


# =============================================================================
# Force Field Differentiability Tests
# =============================================================================

class TestFieldDifferentiability:
    """Test that all force fields are differentiable."""
    
    def test_wind_field_differentiable_wrt_position(self):
        """Test WindField gradient w.r.t. position."""
        field = WindField.create(
            center=(0.0, 0.0),
            size=(2.0, 2.0),
            direction=(1.0, 0.0),
            strength=5.0
        )
        
        def force_magnitude(pos):
            force = field.compute_force(pos, jnp.zeros(2))
            return jnp.sum(force ** 2)
        
        position = jnp.array([0.5, 0.0])
        grad_fn = grad(force_magnitude)
        gradient = grad_fn(position)
        
        assert jnp.all(jnp.isfinite(gradient)), "WindField gradient should be finite"
    
    def test_wind_field_differentiable_wrt_strength(self):
        """Test WindField gradient w.r.t. strength parameter."""
        def make_wind_and_compute(strength):
            field = WindField(
                center=jnp.array([0.0, 0.0]),
                size=jnp.array([2.0, 2.0]),
                direction=jnp.array([1.0, 0.0]),
                strength=strength,
                softness=0.1
            )
            position = jnp.array([0.0, 0.0])
            velocity = jnp.zeros(2)
            force = field.compute_force(position, velocity)
            return jnp.sum(force ** 2)
        
        grad_fn = grad(make_wind_and_compute)
        gradient = grad_fn(5.0)
        
        assert jnp.isfinite(gradient), "Gradient w.r.t. strength should be finite"
        assert gradient > 0, "Increasing strength should increase force magnitude"
    
    def test_vortex_field_differentiable(self):
        """Test VortexField differentiability."""
        field = VortexField.create(
            center=(0.0, 0.0),
            strength=5.0,
            radius=2.0,
            falloff="gaussian"
        )
        
        def force_magnitude(pos):
            force = field.compute_force(pos, jnp.zeros(2))
            return jnp.sum(force ** 2)
        
        position = jnp.array([1.0, 0.0])
        grad_fn = grad(force_magnitude)
        gradient = grad_fn(position)
        
        assert jnp.all(jnp.isfinite(gradient)), "VortexField gradient should be finite"
    
    def test_point_force_differentiable(self):
        """Test PointForce differentiability, especially near center."""
        field = PointForce.create(
            center=(0.0, 0.0),
            strength=10.0,
            falloff_power=2.0,
            softness=0.1  # Softening prevents singularity
        )
        
        def force_magnitude(pos):
            force = field.compute_force(pos, jnp.zeros(2))
            return jnp.sum(force ** 2)
        
        # Test at various distances, including close to center
        for dist in [0.05, 0.1, 0.5, 1.0, 2.0]:
            position = jnp.array([dist, 0.0])
            grad_fn = grad(force_magnitude)
            gradient = grad_fn(position)
            
            assert jnp.all(jnp.isfinite(gradient)), \
                f"PointForce gradient should be finite at distance {dist}"
    
    def test_gravity_well_differentiable(self):
        """Test GravityWell differentiability."""
        field = GravityWell.create(
            center=(0.0, 0.0),
            strength=10.0,
            sigma=1.0
        )
        
        def force_magnitude(pos):
            force = field.compute_force(pos, jnp.zeros(2))
            return jnp.sum(force ** 2)
        
        position = jnp.array([0.5, 0.5])
        grad_fn = grad(force_magnitude)
        gradient = grad_fn(position)
        
        assert jnp.all(jnp.isfinite(gradient)), "GravityWell gradient should be finite"
    
    def test_drag_zone_differentiable(self):
        """Test DragZone differentiability w.r.t. position and velocity."""
        field = DragZone.create(
            center=(0.0, 0.0),
            radius=2.0,
            drag_coefficient=1.0
        )
        
        def force_magnitude(pos, vel):
            force = field.compute_force(pos, vel)
            return jnp.sum(force ** 2)
        
        position = jnp.array([0.5, 0.5])
        velocity = jnp.array([1.0, 0.0])
        
        grad_pos = grad(force_magnitude, argnums=0)(position, velocity)
        grad_vel = grad(force_magnitude, argnums=1)(position, velocity)
        
        assert jnp.all(jnp.isfinite(grad_pos)), "DragZone position gradient should be finite"
        assert jnp.all(jnp.isfinite(grad_vel)), "DragZone velocity gradient should be finite"


# =============================================================================
# Simulation Differentiability Tests
# =============================================================================

class TestSimulationDifferentiability:
    """Test that full simulation is differentiable."""
    
    def test_single_step_differentiable(self, initial_state, simple_config):
        """Test that a single integration step is differentiable."""
        field = WindField.create(
            center=(1.0, 0.0),
            size=(2.0, 2.0),
            direction=(0.0, 1.0),
            strength=5.0
        )
        
        def step_fn(pos):
            state = PhysicsState(
                position=pos,
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            new_state = semi_implicit_euler_step(state, [field], simple_config)
            return jnp.sum(new_state.position ** 2)
        
        position = jnp.array([0.0, 0.0])
        grad_fn = grad(step_fn)
        gradient = grad_fn(position)
        
        assert jnp.all(jnp.isfinite(gradient)), "Single step gradient should be finite"
    
    def test_full_trajectory_differentiable_wrt_initial_position(
        self, initial_state, simple_config
    ):
        """Test gradient of trajectory w.r.t. initial position."""
        field = WindField.create(
            center=(1.0, 0.0),
            size=(2.0, 2.0),
            direction=(0.0, 1.0),
            strength=5.0
        )
        
        def trajectory_endpoint(init_pos):
            state = PhysicsState(
                position=init_pos,
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            positions = simulate_positions_only(state, [field], simple_config)
            return jnp.sum(positions[-1] ** 2)  # Squared distance of endpoint
        
        init_pos = jnp.array([0.0, 0.0])
        grad_fn = grad(trajectory_endpoint)
        gradient = grad_fn(init_pos)
        
        assert jnp.all(jnp.isfinite(gradient)), \
            "Trajectory gradient w.r.t. initial position should be finite"
    
    def test_full_trajectory_differentiable_wrt_initial_velocity(
        self, initial_state, simple_config
    ):
        """Test gradient of trajectory w.r.t. initial velocity."""
        field = WindField.create(
            center=(1.0, 0.0),
            size=(2.0, 2.0),
            direction=(0.0, 1.0),
            strength=5.0
        )
        
        def trajectory_endpoint(init_vel):
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=init_vel,
                time=0.0
            )
            positions = simulate_positions_only(state, [field], simple_config)
            return jnp.sum(positions[-1] ** 2)
        
        init_vel = jnp.array([1.0, 0.0])
        grad_fn = grad(trajectory_endpoint)
        gradient = grad_fn(init_vel)
        
        assert jnp.all(jnp.isfinite(gradient)), \
            "Trajectory gradient w.r.t. initial velocity should be finite"
    
    def test_trajectory_differentiable_wrt_field_strength(self, simple_config):
        """Test gradient of trajectory w.r.t. field strength."""
        def trajectory_with_strength(strength):
            field = WindField(
                center=jnp.array([1.0, 0.0]),
                size=jnp.array([2.0, 2.0]),
                direction=jnp.array([0.0, 1.0]),
                strength=strength,
                softness=0.1
            )
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            positions = simulate_positions_only(state, [field], simple_config)
            # Return y-coordinate of endpoint (should increase with wind strength)
            return positions[-1, 1]
        
        grad_fn = grad(trajectory_with_strength)
        gradient = grad_fn(5.0)
        
        assert jnp.isfinite(gradient), "Gradient w.r.t. field strength should be finite"
        assert gradient > 0, "Increasing wind strength should increase y displacement"
    
    def test_trajectory_differentiable_wrt_field_position(self, simple_config):
        """Test gradient of trajectory w.r.t. field center position."""
        def trajectory_with_field_center(field_center):
            field = WindField(
                center=field_center,
                size=jnp.array([2.0, 2.0]),
                direction=jnp.array([0.0, 1.0]),
                strength=5.0,
                softness=0.1
            )
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            positions = simulate_positions_only(state, [field], simple_config)
            return positions[-1, 1]  # y-coordinate of endpoint
        
        field_center = jnp.array([1.0, 0.0])
        grad_fn = grad(trajectory_with_field_center)
        gradient = grad_fn(field_center)
        
        assert jnp.all(jnp.isfinite(gradient)), \
            "Gradient w.r.t. field center should be finite"


# =============================================================================
# Loss Function Differentiability Tests
# =============================================================================

class TestLossDifferentiability:
    """Test that loss functions are differentiable."""
    
    def test_trajectory_loss_differentiable(self):
        """Test gradient of trajectory loss."""
        target = generate_line_trajectory((0, 0), (2, 1), num_points=50)
        
        def loss_fn(predicted):
            return trajectory_loss_same_length(predicted, target)
        
        predicted = generate_line_trajectory((0, 0), (2, 0.5), num_points=50)
        grad_fn = grad(loss_fn)
        gradient = grad_fn(predicted)
        
        assert jnp.all(jnp.isfinite(gradient)), "Loss gradient should be finite"
        assert gradient.shape == predicted.shape, "Gradient shape should match input"
    
    def test_end_to_end_loss_gradient(self, simple_config):
        """Test full pipeline: field params -> simulation -> loss -> gradient."""
        target = generate_arc_trajectory(
            center=(2.0, 0.0),
            radius=1.0,
            start_angle=jnp.pi,
            end_angle=0,
            num_points=simple_config.num_steps
        )
        
        def full_loss(field_strength):
            field = WindField(
                center=jnp.array([1.0, 0.0]),
                size=jnp.array([3.0, 3.0]),
                direction=jnp.array([0.0, 1.0]),
                strength=field_strength,
                softness=0.1
            )
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            predicted = simulate_positions_only(state, [field], simple_config)
            return trajectory_loss_same_length(predicted, target)
        
        grad_fn = grad(full_loss)
        gradient = grad_fn(5.0)
        
        assert jnp.isfinite(gradient), "End-to-end gradient should be finite"


# =============================================================================
# Numerical Gradient Verification
# =============================================================================

class TestNumericalGradients:
    """Verify analytical gradients match numerical gradients."""
    
    def test_wind_force_gradient_matches_numerical(self):
        """Compare analytical gradient to finite difference."""
        field = WindField.create(
            center=(0.0, 0.0),
            size=(2.0, 2.0),
            direction=(1.0, 0.0),
            strength=5.0
        )
        
        def force_x(pos):
            force = field.compute_force(pos, jnp.zeros(2))
            return force[0]
        
        position = jnp.array([0.5, 0.0])
        
        # Analytical gradient
        analytical = grad(force_x)(position)
        
        # Numerical gradient (finite difference)
        eps = 1e-5
        numerical = jnp.zeros(2)
        for i in range(2):
            pos_plus = position.at[i].add(eps)
            pos_minus = position.at[i].add(-eps)
            numerical = numerical.at[i].set(
                (force_x(pos_plus) - force_x(pos_minus)) / (2 * eps)
            )
        
        # Should be close (relaxed tolerance for soft boundary effects)
        assert jnp.allclose(analytical, numerical, atol=1e-2), \
            f"Analytical {analytical} != Numerical {numerical}"
    
    def test_simulation_gradient_matches_numerical(self, simple_config):
        """Compare simulation gradient to finite difference."""
        def trajectory_endpoint_y(init_vel_y):
            field = WindField.create(
                center=(1.0, 0.0),
                size=(2.0, 2.0),
                direction=(0.0, 1.0),
                strength=5.0
            )
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=jnp.array([1.0, init_vel_y]),
                time=0.0
            )
            positions = simulate_positions_only(state, [field], simple_config)
            return positions[-1, 1]
        
        init_vel_y = 0.0
        
        # Analytical gradient
        analytical = grad(trajectory_endpoint_y)(init_vel_y)
        
        # Numerical gradient
        eps = 1e-5
        numerical = (
            trajectory_endpoint_y(init_vel_y + eps) - 
            trajectory_endpoint_y(init_vel_y - eps)
        ) / (2 * eps)
        
        # Allow some tolerance due to numerical integration
        assert jnp.abs(analytical - numerical) < 0.1, \
            f"Analytical {analytical} != Numerical {numerical}"


# =============================================================================
# Long Simulation Tests
# =============================================================================

class TestLongSimulations:
    """Test gradient stability for longer simulations."""
    
    def test_gradient_stability_500_steps(self):
        """Test that gradients remain stable over 500 steps."""
        config = SimulationConfig(
            dt=0.01,
            num_steps=500,
            ball_mass=1.0,
            bounds=((-10.0, 10.0), (-10.0, 10.0)),
            boundary_softness=0.2,
            boundary_strength=20.0,
        )
        
        def trajectory_loss_fn(field_strength):
            field = WindField(
                center=jnp.array([2.0, 0.0]),
                size=jnp.array([4.0, 4.0]),
                direction=jnp.array([0.0, 1.0]),
                strength=field_strength,
                softness=0.1
            )
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            positions = simulate_positions_only(state, [field], config)
            # Loss: distance from target endpoint
            target = jnp.array([5.0, 2.0])
            return jnp.sum((positions[-1] - target) ** 2)
        
        # Compute gradient
        grad_fn = grad(trajectory_loss_fn)
        gradient = grad_fn(5.0)
        
        assert jnp.isfinite(gradient), "Gradient should be finite for 500-step simulation"
        assert jnp.abs(gradient) < 1e6, "Gradient should not explode"


# =============================================================================
# Multiple Fields Tests
# =============================================================================

class TestMultipleFields:
    """Test differentiability with multiple interacting fields."""
    
    def test_multiple_fields_gradient(self, simple_config):
        """Test gradient with multiple fields."""
        def multi_field_loss(strengths):
            wind = WindField(
                center=jnp.array([1.0, 0.0]),
                size=jnp.array([2.0, 2.0]),
                direction=jnp.array([0.0, 1.0]),
                strength=strengths[0],
                softness=0.1
            )
            vortex = VortexField(
                center=jnp.array([2.0, 1.0]),
                strength=strengths[1],
                radius=1.5,
                falloff="gaussian",
                softness=0.1
            )
            attractor = PointForce(
                center=jnp.array([3.0, 0.0]),
                strength=strengths[2],
                falloff_power=2.0,
                max_distance=5.0,
                softness=0.2
            )
            
            state = PhysicsState(
                position=jnp.array([0.0, 0.0]),
                velocity=jnp.array([1.0, 0.0]),
                time=0.0
            )
            positions = simulate_positions_only(
                state, [wind, vortex, attractor], simple_config
            )
            
            target = jnp.array([3.0, 1.0])
            return jnp.sum((positions[-1] - target) ** 2)
        
        strengths = jnp.array([5.0, 3.0, 2.0])
        grad_fn = grad(multi_field_loss)
        gradient = grad_fn(strengths)
        
        assert jnp.all(jnp.isfinite(gradient)), \
            "Gradient should be finite for multiple fields"
        assert gradient.shape == (3,), "Should have gradient for each field strength"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

