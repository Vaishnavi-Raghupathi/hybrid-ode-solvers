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

# Define Correction Neural Network - DEEPER for better learning
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

# Simple PINN
class SimplePINN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )
        
    def forward(self, t):
        return self.net(t)

def train_pinn(t_true, y_true, epochs=500, lr=0.001):
    """Train PINN on trajectory data"""
    model = SimplePINN(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    t_tensor = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32)
    
    for epoch in range(epochs):
        pred = model(t_tensor)
        loss = criterion(pred, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"PINN Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
    
    return model

# Run Full Experiment
def run_experiment(mu=1.0, dt=0.05, t_max=20.0, n_samples=20000):
    """Complete experimental pipeline"""
    
    print("=" * 60)
    print("HYBRID ODE SOLVER EXPERIMENT")
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
    
    # Step 4: Train PINN baseline
    print("\n[4/6] Training PINN baseline...")
    pinn = train_pinn(t_true, y_true, epochs=500)
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
    
    # PINN
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
    print(f"  RK4:    {error_rk4[-1]:.6f}")
    print(f"  Hybrid: {error_hybrid[-1]:.6f}")
    print(f"  PINN:   {error_pinn[-1]:.6f}")
    
    if error_hybrid[-1] < error_rk4[-1]:
        improvement = (1 - error_hybrid[-1]/error_rk4[-1])*100
        print(f"\n✓ Hybrid improvement: {improvement:.1f}% better than RK4")
    else:
        degradation = (error_hybrid[-1]/error_rk4[-1] - 1)*100
        print(f"\n⚠ Hybrid degradation: {degradation:.1f}% worse than RK4")
    
    # Print mean errors for better comparison
    print(f"\nMean Error over trajectory:")
    print(f"  RK4:    {np.mean(error_rk4):.6f}")
    print(f"  Hybrid: {np.mean(error_hybrid):.6f}")
    print(f"  PINN:   {np.mean(error_pinn):.6f}")
    
    return {
        't_true': t_true,
        'y_true': y_true,
        'y_rk4': y_rk4,
        'y_hybrid': y_hybrid,
        'y_pinn': y_pinn,
        'error_rk4': error_rk4,
        'error_hybrid': error_hybrid,
        'error_pinn': error_pinn
    }

# Visualize Results
def plot_results(results):
    """Create publication-quality plots"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Error vs Time
    ax = axes[0]
    ax.plot(results['t_true'], results['error_rk4'], 'r-', label='RK4', linewidth=2, alpha=0.8)
    ax.plot(results['t_true'], results['error_hybrid'], 'g-', label='Hybrid (RK4 + NN)', linewidth=2, alpha=0.8)
    ax.plot(results['t_true'], results['error_pinn'], 'orange', label='PINN', linewidth=2, alpha=0.8)
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
    ax.plot(results['t_true'], results['y_pinn'][:, 0], 'orange', label='PINN', linewidth=2, alpha=0.6)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('y₁', fontsize=12)
    ax.set_title('Trajectory Comparison (y₁ component)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_solver_results_better.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved figure as 'hybrid_solver_results_better.png'")
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