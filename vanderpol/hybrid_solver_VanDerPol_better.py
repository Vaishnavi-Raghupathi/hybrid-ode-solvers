import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)
torch.manual_seed(42)

# Define the Van der Pol ODE
def vdp_rhs(t, y, mu=1.0):
    """Van der Pol oscillator dynamics"""
    y1, y2 = y
    dy1 = y2
    dy2 = mu * (1 - y1**2) * y2 - y1
    return np.array([dy1, dy2])

# Implement RK4 Solver
def rk4_step(f, t, y, dt, mu=1.0):
    """Single RK4 step"""
    k1 = f(t, y, mu)
    k2 = f(t + dt/2, y + dt/2 * k1, mu)
    k3 = f(t + dt/2, y + dt/2 * k2, mu)
    k4 = f(t + dt, y + dt * k3, mu)
    y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

# Generate Ground Truth
def generate_ground_truth(mu=1.0, dt=0.05, t_max=10.0, y0=None):
    """Generate high-accuracy reference solution"""
    if y0 is None:
        y0 = np.array([2.0, 0.0])
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max + dt, dt)
    
    sol = solve_ivp(
        lambda t, y: vdp_rhs(t, y, mu),
        t_span, y0, t_eval=t_eval,
        method='DOP853',  # Higher order method for ground truth
        rtol=1e-12, atol=1e-12
    )
    return sol.t, sol.y.T

# Generate Training Dataset WITH MULTIPLE TRAJECTORIES
def generate_training_data_multi_trajectory(mu=1.0, dt=0.05, t_max=10.0, n_samples=5000):
    """Generate dataset from MULTIPLE initial conditions"""
    dataset = []
    n_trajectories = 15  # More trajectories for better generalization
    samples_per_traj = n_samples // n_trajectories
    
    for traj_idx in range(n_trajectories):
        # Random initial conditions - wider range
        y0 = np.random.uniform(-2.5, 2.5, size=2)
        
        # Generate ground truth for this trajectory
        t_true, y_true = generate_ground_truth(mu, dt, t_max, y0)
        
        # Sample from this trajectory
        for _ in range(samples_per_traj):
            idx = np.random.randint(0, len(t_true) - 1)
            
            t_curr = t_true[idx]
            y_curr = y_true[idx]
            y_next_true = y_true[idx + 1]
            
            # RK4 prediction
            y_next_rk4 = rk4_step(vdp_rhs, t_curr, y_curr, dt, mu)
            
            # Local error
            delta = y_next_true - y_next_rk4
            
            dataset.append({
                'input': np.array([t_curr, y_curr[0], y_curr[1], dt]),
                'target': delta
            })
    
    return dataset

# Define Correction Neural Network
class CorrectionNetwork(nn.Module):
    """Correction network with learnable output scaling"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 2)
        )
        
    def forward(self, x):
        return self.net(x)

# Train Correction Network
def train_correction_network(dataset, epochs=300, batch_size=128, lr=0.001):
    """Train with careful regularization"""
    model = CorrectionNetwork(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    criterion = nn.MSELoss()
    
    # Convert dataset efficiently
    inputs = np.array([d['input'] for d in dataset])
    targets = np.array([d['target'] for d in dataset])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(len(inputs))
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]
        
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(inputs), batch_size):
            batch_input = inputs_shuffled[i:i+batch_size]
            batch_target = targets_shuffled[i:i+batch_size]
            
            pred = model(batch_input)
            loss = criterion(pred, batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 30 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.8f}")
    
    return model, losses

# ============================================================================
# TRUE PHYSICS-INFORMED NEURAL NETWORK (PINN)
# ============================================================================

class TruePINN(nn.Module):
    """
    Physics-Informed Neural Network for Van der Pol oscillator
    Network learns y(t) while respecting the physics (ODE)
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # Neural network: t -> [y1, y2]
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)  # Output: [y1, y2]
        )
    
    def forward(self, t):
        """Forward pass through network"""
        return self.net(t)


def compute_pinn_physics_loss(model, t_physics, mu=1.0):
    """
    THE KEY PART: Compute how well the network satisfies the Van der Pol ODE
    
    Van der Pol equations:
    - dy1/dt = y2
    - dy2/dt = mu*(1 - y1^2)*y2 - y1
    
    We penalize deviations from these equations!
    """
    # Enable gradient computation for input
    t_physics.requires_grad = True
    
    # Get network prediction
    y_pred = model(t_physics)
    y1 = y_pred[:, 0:1]  # Position
    y2 = y_pred[:, 1:2]  # Velocity
    
    # Compute derivatives using automatic differentiation
    dy1_dt = torch.autograd.grad(
        outputs=y1, 
        inputs=t_physics,
        grad_outputs=torch.ones_like(y1),
        create_graph=True,
        retain_graph=True
    )[0]
    
    dy2_dt = torch.autograd.grad(
        outputs=y2,
        inputs=t_physics,
        grad_outputs=torch.ones_like(y2),
        create_graph=True
    )[0]
    
    # Physics residuals (how much we violate the ODE)
    # Equation 1: dy1/dt - y2 = 0
    physics_residual_1 = dy1_dt - y2
    
    # Equation 2: dy2/dt - mu*(1 - y1^2)*y2 + y1 = 0
    physics_residual_2 = dy2_dt - mu * (1 - y1**2) * y2 + y1
    
    # We want these residuals to be ZERO (perfect physics satisfaction)
    physics_loss = torch.mean(physics_residual_1**2) + torch.mean(physics_residual_2**2)
    
    return physics_loss


def compute_pinn_data_loss(model, t_data, y_data):
    """
    Data loss: How well does network fit the observed data points?
    """
    y_pred = model(t_data)
    data_loss = torch.mean((y_pred - y_data)**2)
    return data_loss


def compute_pinn_ic_loss(model, t0, y0):
    """
    Initial condition loss: Network should match initial conditions exactly
    """
    y_pred = model(t0)
    ic_loss = torch.mean((y_pred - y0)**2)
    return ic_loss


def train_true_pinn(t_data, y_data, mu=1.0, epochs=2000, lr=0.001,
                    lambda_data=1.0, lambda_physics=0.01, lambda_ic=10.0):
    """
    Train PINN with three loss components:
    1. Data loss: Fit the observed data
    2. Physics loss: Satisfy the Van der Pol ODE
    3. Initial condition loss: Match initial conditions
    """
    model = TruePINN(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert data to tensors
    t_data_tensor = torch.tensor(t_data.reshape(-1, 1), dtype=torch.float32)
    y_data_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Initial conditions
    t0 = torch.tensor([[0.0]], dtype=torch.float32)
    y0 = torch.tensor([[y_data[0, 0], y_data[0, 1]]], dtype=torch.float32)
    
    # Physics collocation points (many points where we enforce ODE)
    n_physics = 1000
    t_physics_np = np.linspace(t_data[0], t_data[-1], n_physics)
    t_physics = torch.tensor(t_physics_np.reshape(-1, 1), dtype=torch.float32)
    
    losses_total = []
    losses_data = []
    losses_physics = []
    losses_ic = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute all loss components
        loss_data = compute_pinn_data_loss(model, t_data_tensor, y_data_tensor)
        loss_physics = compute_pinn_physics_loss(model, t_physics, mu)
        loss_ic = compute_pinn_ic_loss(model, t0, y0)
        
        # Weighted combination
        total_loss = (lambda_data * loss_data + 
                     lambda_physics * loss_physics + 
                     lambda_ic * loss_ic)
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        losses_total.append(total_loss.item())
        losses_data.append(loss_data.item())
        losses_physics.append(loss_physics.item())
        losses_ic.append(loss_ic.item())
        
        if epoch % 200 == 0:
            print(f"PINN Epoch {epoch:4d}/{epochs} | Total: {total_loss.item():.6f} | "
                  f"Data: {loss_data.item():.6f} | Physics: {loss_physics.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")
    
    return model, (losses_total, losses_data, losses_physics, losses_ic)


# Run Full Experiment
def run_experiment(mu=1.0, dt=0.05, t_max=20.0, n_samples=20000):
    """Complete experimental pipeline"""
    
    print("=" * 60)
    print("HYBRID ODE SOLVER EXPERIMENT WITH TRUE PINN")
    print("=" * 60)
    
    # Step 1: Generate ground truth (for testing)
    print("\n[1/6] Generating ground truth...")
    t_true, y_true = generate_ground_truth(mu, dt, t_max)
    print(f"Generated {len(t_true)} time points")
    
    # Step 2: Generate training data from MULTIPLE trajectories
    print("\n[2/6] Generating training dataset from multiple trajectories...")
    dataset = generate_training_data_multi_trajectory(mu, dt, t_max, n_samples)
    print(f"Generated {len(dataset)} training samples")
    
    # Step 3: Train correction network
    print("\n[3/6] Training correction network...")
    corr_net, losses = train_correction_network(dataset, epochs=300, batch_size=128)
    print(f"Training complete. Final loss: {losses[-1]:.8f}")
    
    # Step 4: Train TRUE PINN
    print("\n[4/6] Training TRUE PINN (Physics-Informed)...")
    # Use sparse data for PINN (every 10th point to make it challenging)
    t_pinn_data = t_true[::10]
    y_pinn_data = y_true[::10]
    print(f"PINN training with {len(t_pinn_data)} sparse data points")
    pinn, pinn_losses = train_true_pinn(
        t_pinn_data, y_pinn_data, mu=mu, epochs=2000, lr=0.001,
        lambda_data=1.0, lambda_physics=0.01, lambda_ic=10.0
    )
    print("PINN training complete")
    
    # Step 5: Rollout all methods
    print("\n[5/6] Rolling out all methods...")
    y0 = np.array([2.0, 0.0])
    n_steps = len(t_true) - 1
    
    # RK4 only
    y_rk4 = [y0.copy()]
    for i in range(n_steps):
        y_next = rk4_step(vdp_rhs, i*dt, y_rk4[-1], dt, mu)
        y_rk4.append(y_next)
    y_rk4 = np.array(y_rk4)
    
    # Hybrid (RK4 + Correction)
    y_hybrid = [y0.copy()]
    corr_net.eval()
    for i in range(n_steps):
        y_curr = y_hybrid[-1]
        t_curr = i * dt
        
        # RK4 step
        y_rk4_pred = rk4_step(vdp_rhs, t_curr, y_curr, dt, mu)
        
        # Neural correction
        with torch.no_grad():
            inp = torch.tensor([t_curr, y_curr[0], y_curr[1], dt], dtype=torch.float32)
            correction = corr_net(inp).numpy()
        
        # Apply correction
        y_next = y_rk4_pred + correction
        y_hybrid.append(y_next)
    y_hybrid = np.array(y_hybrid)
    
    # TRUE PINN - direct prediction
    pinn.eval()
    with torch.no_grad():
        t_pinn = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32)
        y_pinn = pinn(t_pinn).numpy()
    
    # Step 6: Calculate errors
    print("\n[6/6] Calculating errors...")
    error_rk4 = np.linalg.norm(y_rk4 - y_true, axis=1)
    error_hybrid = np.linalg.norm(y_hybrid - y_true, axis=1)
    error_pinn = np.linalg.norm(y_pinn - y_true, axis=1)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final Error (t={t_max}):")
    print(f"  RK4:         {error_rk4[-1]:.6f}")
    print(f"  Hybrid:      {error_hybrid[-1]:.6f}")
    print(f"  TRUE PINN:   {error_pinn[-1]:.6f}")
    
    if error_hybrid[-1] < error_rk4[-1]:
        improvement = (1 - error_hybrid[-1]/error_rk4[-1])*100
        print(f"\n✓ Hybrid improvement: {improvement:.1f}% better than RK4")
    else:
        degradation = (error_hybrid[-1]/error_rk4[-1] - 1)*100
        print(f"\n⚠ Hybrid degradation: {degradation:.1f}% worse than RK4")
    
    if error_pinn[-1] < error_rk4[-1]:
        improvement = (1 - error_pinn[-1]/error_rk4[-1])*100
        print(f"✓ PINN improvement: {improvement:.1f}% better than RK4")
    else:
        degradation = (error_pinn[-1]/error_rk4[-1] - 1)*100
        print(f"⚠ PINN degradation: {degradation:.1f}% worse than RK4")
    
    # Print mean errors for better comparison
    print(f"\nMean Error over trajectory:")
    print(f"  RK4:         {np.mean(error_rk4):.6f}")
    print(f"  Hybrid:      {np.mean(error_hybrid):.6f}")
    print(f"  TRUE PINN:   {np.mean(error_pinn):.6f}")
    
    return {
        't_true': t_true,
        'y_true': y_true,
        'y_rk4': y_rk4,
        'y_hybrid': y_hybrid,
        'y_pinn': y_pinn,
        'error_rk4': error_rk4,
        'error_hybrid': error_hybrid,
        'error_pinn': error_pinn,
        'pinn_losses': pinn_losses
    }

# Visualize Results
def plot_results(results):
    """Create publication-quality plots"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Error vs Time
    ax = axes[0]
    ax.plot(results['t_true'], results['error_rk4'], 'r-', label='RK4', linewidth=2, alpha=0.8)
    ax.plot(results['t_true'], results['error_hybrid'], 'g-', label='Hybrid (RK4 + NN)', linewidth=2, alpha=0.8)
    ax.plot(results['t_true'], results['error_pinn'], 'b-', label='TRUE PINN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Error Evolution Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Trajectory Comparison
    ax = axes[1]
    ax.plot(results['t_true'], results['y_true'][:, 0], 'k--', label='Ground Truth', linewidth=3, alpha=0.7)
    ax.plot(results['t_true'], results['y_rk4'][:, 0], 'r-', label='RK4', linewidth=2, alpha=0.6)
    ax.plot(results['t_true'], results['y_hybrid'][:, 0], 'g-', label='Hybrid', linewidth=2, alpha=0.8)
    ax.plot(results['t_true'], results['y_pinn'][:, 0], 'b-', label='TRUE PINN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('y₁', fontsize=12)
    ax.set_title('Trajectory Comparison (y₁ component)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: PINN Training Losses
    ax = axes[2]
    losses_total, losses_data, losses_physics, losses_ic = results['pinn_losses']
    epochs = np.arange(len(losses_total))
    ax.plot(epochs, losses_total, 'k-', label='Total Loss', linewidth=2)
    ax.plot(epochs, losses_data, 'b-', label='Data Loss', linewidth=1.5, alpha=0.7)
    ax.plot(epochs, losses_physics, 'r-', label='Physics Loss', linewidth=1.5, alpha=0.7)
    ax.plot(epochs, losses_ic, 'g-', label='IC Loss', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('PINN Training: Loss Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('hybrid_solver_with_true_pinn.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved figure as 'hybrid_solver_with_true_pinn.png'")
    plt.show()

# Run everything
if __name__ == "__main__":
    results = run_experiment(
        mu=2.5,           
        dt=0.25,          
        t_max=20.0,       
        n_samples=25000   
    )
    
    plot_results(results)