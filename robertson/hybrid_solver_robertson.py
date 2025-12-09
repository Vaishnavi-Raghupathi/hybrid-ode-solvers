import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)
torch.manual_seed(42)

# Define the Robertson Problem (Stiff ODE System)
def robertson_rhs(t, y):
    """
    Robertson chemical reaction system (stiff ODE)
    dy1/dt = -0.04*y1 + 1e4*y2*y3
    dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    dy3/dt = 3e7*y2^2
    
    Conservation: y1 + y2 + y3 = 1
    """
    y1, y2, y3 = y
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3 = 3e7 * y2**2
    return np.array([dy1, dy2, dy3])

# Implement RK4 Solver
def rk4_step(f, t, y, dt):
    """Single RK4 step"""
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

# Generate Ground Truth
def generate_ground_truth(dt=0.01, t_max=10.0, y0=None):
    """Generate high-accuracy reference solution using stiff solver"""
    if y0 is None:
        y0 = np.array([1.0, 0.0, 0.0])  # Initial condition for Robertson
    
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max + dt, dt)
    
    # Use stiff solver for ground truth
    sol = solve_ivp(
        robertson_rhs,
        t_span, y0, t_eval=t_eval,
        method='Radau',  # Stiff solver
        rtol=1e-10, atol=1e-12
    )
    return sol.t, sol.y.T

# Generate Training Dataset WITH MULTIPLE TRAJECTORIES
def generate_training_data_multi_trajectory(dt=0.01, t_max=10.0, n_samples=10000):
    """Generate dataset from MULTIPLE initial conditions"""
    dataset = []
    n_trajectories = 20  # Multiple trajectories for better generalization
    samples_per_traj = n_samples // n_trajectories
    
    for traj_idx in range(n_trajectories):
        # Random initial conditions that satisfy conservation law
        # y1 + y2 + y3 = 1
        r1 = np.random.uniform(0.5, 1.0)
        r2 = np.random.uniform(0.0, (1 - r1))
        y0 = np.array([r1, r2, 1 - r1 - r2])
        
        # Generate ground truth for this trajectory
        try:
            t_true, y_true = generate_ground_truth(dt, t_max, y0)
        except:
            continue  # Skip if solver fails
        
        # Sample from this trajectory
        for _ in range(samples_per_traj):
            idx = np.random.randint(0, len(t_true) - 1)
            
            t_curr = t_true[idx]
            y_curr = y_true[idx]
            y_next_true = y_true[idx + 1]
            
            # RK4 prediction
            y_next_rk4 = rk4_step(robertson_rhs, t_curr, y_curr, dt)
            
            # Clip to maintain physical bounds
            y_next_rk4 = np.clip(y_next_rk4, 0, 1)
            
            # Local error
            delta = y_next_true - y_next_rk4
            
            dataset.append({
                'input': np.array([t_curr, y_curr[0], y_curr[1], y_curr[2], dt]),
                'target': delta
            })
    
    return dataset

# Define Correction Neural Network - DEEPER for better learning
class CorrectionNetwork(nn.Module):
    """Correction network for Robertson problem"""
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),  # Input: [t, y1, y2, y3, dt]
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 3)  # Output: [delta_y1, delta_y2, delta_y3]
        )
        
    def forward(self, x):
        return self.net(x)

# Train Correction Network
def train_correction_network(dataset, epochs=400, batch_size=256, lr=0.001):
    """Train with careful regularization"""
    model = CorrectionNetwork(hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5)
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
        
        if epoch % 40 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model, losses

# Simple PINN
class SimplePINN(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3)  # Output: [y1, y2, y3]
        )
        
    def forward(self, t):
        return self.net(t)

def train_pinn(t_true, y_true, epochs=1000, lr=0.001):
    """Train PINN on trajectory data"""
    model = SimplePINN(hidden_size=128)
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
        
        if epoch % 100 == 0:
            print(f"PINN Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
    
    return model

# Run Full Experiment
def run_experiment(dt=0.01, t_max=10.0, n_samples=30000):
    """Complete experimental pipeline for Robertson problem"""
    
    print("=" * 60)
    print("HYBRID ODE SOLVER - ROBERTSON STIFF PROBLEM")
    print("=" * 60)
    print(f"Problem: Stiff chemical kinetics (3 species)")
    print(f"Time step: {dt}, Max time: {t_max}")
    print("=" * 60)
    
    # Step 1: Generate ground truth (for testing)
    print("\n[1/6] Generating ground truth...")
    t_true, y_true = generate_ground_truth(dt, t_max)
    print(f"Generated {len(t_true)} time points")
    print(f"Conservation check: y1+y2+y3 = {y_true[-1].sum():.6f} (should be ~1.0)")
    
    # Step 2: Generate training data from MULTIPLE trajectories
    print("\n[2/6] Generating training dataset from multiple trajectories...")
    dataset = generate_training_data_multi_trajectory(dt, t_max, n_samples)
    print(f"Generated {len(dataset)} training samples")
    
    # Step 3: Train correction network
    print("\n[3/6] Training correction network...")
    corr_net, losses = train_correction_network(dataset, epochs=400, batch_size=256)
    print(f"Training complete. Final loss: {losses[-1]:.8f}")
    
    # Step 4: Train PINN baseline
    print("\n[4/6] Training PINN baseline...")
    pinn = train_pinn(t_true, y_true, epochs=1000)
    print("PINN training complete")
    
    # Step 5: Rollout all methods
    print("\n[5/6] Rolling out all methods...")
    y0 = np.array([1.0, 0.0, 0.0])  # Standard Robertson initial condition
    n_steps = len(t_true) - 1
    
    # RK4 only
    y_rk4 = [y0.copy()]
    for i in range(n_steps):
        y_next = rk4_step(robertson_rhs, i*dt, y_rk4[-1], dt)
        # Clip to maintain physical bounds
        y_next = np.clip(y_next, 0, 1)
        y_rk4.append(y_next)
    y_rk4 = np.array(y_rk4)
    
    # Hybrid (RK4 + Correction)
    y_hybrid = [y0.copy()]
    corr_net.eval()
    for i in range(n_steps):
        y_curr = y_hybrid[-1]
        t_curr = i * dt
        
        # RK4 step
        y_rk4_pred = rk4_step(robertson_rhs, t_curr, y_curr, dt)
        y_rk4_pred = np.clip(y_rk4_pred, 0, 1)
        
        # Neural correction
        with torch.no_grad():
            inp = torch.tensor([t_curr, y_curr[0], y_curr[1], y_curr[2], dt], dtype=torch.float32)
            correction = corr_net(inp).numpy()
            # Scale correction to avoid instability
            correction = np.clip(correction, -0.1, 0.1)
        
        # Apply correction
        y_next = y_rk4_pred + correction
        # Enforce physical constraints
        y_next = np.clip(y_next, 0, 1)
        y_hybrid.append(y_next)
    y_hybrid = np.array(y_hybrid)
    
    # PINN
    pinn.eval()
    with torch.no_grad():
        t_pinn = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32)
        y_pinn = pinn(t_pinn).numpy()
        # Clip PINN output
        y_pinn = np.clip(y_pinn, 0, 1)
    
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
    
    # Conservation check
    print(f"\nConservation (y1+y2+y3 at t={t_max}):")
    print(f"  Truth:  {y_true[-1].sum():.8f}")
    print(f"  RK4:    {y_rk4[-1].sum():.8f}")
    print(f"  Hybrid: {y_hybrid[-1].sum():.8f}")
    print(f"  PINN:   {y_pinn[-1].sum():.8f}")
    
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
    """Create publication-quality plots for Robertson problem"""
    fig = plt.figure(figsize=(14, 10))
    
    # Create 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Error vs Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results['t_true'], results['error_rk4'], 'r-', label='RK4', linewidth=2, alpha=0.8)
    ax1.plot(results['t_true'], results['error_hybrid'], 'g-', label='Hybrid (RK4 + NN)', linewidth=2, alpha=0.8)
    ax1.plot(results['t_true'], results['error_pinn'], 'orange', label='PINN', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Error Evolution Over Time - Robertson Problem', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2-4: Individual species trajectories
    species_names = ['y₁ (Species A)', 'y₂ (Species B)', 'y₃ (Species C)']
    colors_truth = 'black'
    
    for idx, species_name in enumerate(species_names):
        ax = fig.add_subplot(gs[1 + idx//2, idx%2])
        
        ax.plot(results['t_true'], results['y_true'][:, idx], '--', color=colors_truth, 
                label='Ground Truth', linewidth=3, alpha=0.7)
        ax.plot(results['t_true'], results['y_rk4'][:, idx], 'r-', 
                label='RK4', linewidth=2, alpha=0.6)
        ax.plot(results['t_true'], results['y_hybrid'][:, idx], 'g-', 
                label='Hybrid', linewidth=2, alpha=0.8)
        ax.plot(results['t_true'], results['y_pinn'][:, idx], 'orange', 
                label='PINN', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Concentration', fontsize=11)
        ax.set_title(f'{species_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    plt.savefig('robertson_hybrid_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved figure as 'robertson_hybrid_results.png'")
    plt.show()

# Run everything
if __name__ == "__main__":
    results = run_experiment(
        dt=0.05,          
        t_max=10.0,       
        n_samples=30000   
    )
    
    plot_results(results)