import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
torch.manual_seed(42)

# Define the Lorenz System (Chaotic Attractor)
def lorenz_rhs(t, y, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz system - famous chaotic attractor
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    
    Default parameters: σ=10, ρ=28, β=8/3 (chaotic regime)
    """
    x, y, z = y
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

# Implement RK4 Solver
def rk4_step(f, t, y, dt, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Single RK4 step"""
    k1 = f(t, y, sigma, rho, beta)
    k2 = f(t + dt/2, y + dt/2 * k1, sigma, rho, beta)
    k3 = f(t + dt/2, y + dt/2 * k2, sigma, rho, beta)
    k4 = f(t + dt, y + dt * k3, sigma, rho, beta)
    y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

# Generate Ground Truth
def generate_ground_truth(dt=0.01, t_max=20.0, y0=None, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Generate high-accuracy reference solution"""
    if y0 is None:
        y0 = np.array([1.0, 1.0, 1.0])  # Standard Lorenz initial condition
    
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max + dt, dt)
    
    # High accuracy solution
    sol = solve_ivp(
        lambda t, y: lorenz_rhs(t, y, sigma, rho, beta),
        t_span, y0, t_eval=t_eval,
        method='DOP853',  # High-order adaptive method
        rtol=1e-10, atol=1e-12
    )
    return sol.t, sol.y.T

# Generate Training Dataset WITH MULTIPLE TRAJECTORIES
def generate_training_data_multi_trajectory(dt=0.01, t_max=20.0, n_samples=20000, 
                                            sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Generate dataset from MULTIPLE initial conditions"""
    dataset = []
    n_trajectories = 25  # Multiple trajectories from different parts of attractor
    samples_per_traj = n_samples // n_trajectories
    
    for traj_idx in range(n_trajectories):
        # Random initial conditions exploring the attractor
        y0 = np.random.uniform(-15, 15, size=3)
        
        # Generate ground truth for this trajectory
        try:
            t_true, y_true = generate_ground_truth(dt, t_max, y0, sigma, rho, beta)
        except:
            continue  # Skip if solver fails
        
        # Skip first 100 steps to avoid transient behavior
        start_idx = min(100, len(t_true) // 4)
        
        # Sample from this trajectory
        for _ in range(samples_per_traj):
            idx = np.random.randint(start_idx, len(t_true) - 1)
            
            t_curr = t_true[idx]
            y_curr = y_true[idx]
            y_next_true = y_true[idx + 1]
            
            # RK4 prediction
            y_next_rk4 = rk4_step(lorenz_rhs, t_curr, y_curr, dt, sigma, rho, beta)
            
            # Local error
            delta = y_next_true - y_next_rk4
            
            dataset.append({
                'input': np.array([t_curr, y_curr[0], y_curr[1], y_curr[2], dt]),
                'target': delta
            })
    
    return dataset

# Define Correction Neural Network
class CorrectionNetwork(nn.Module):
    """Correction network for Lorenz system"""
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),  # Input: [t, x, y, z, dt]
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 3)  # Output: [delta_x, delta_y, delta_z]
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
            nn.Linear(hidden_size, 3)  # Output: [x, y, z]
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
def run_experiment(dt=0.01, t_max=20.0, n_samples=30000, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Complete experimental pipeline for Lorenz system"""
    
    print("=" * 60)
    print("HYBRID ODE SOLVER - LORENZ CHAOTIC SYSTEM")
    print("=" * 60)
    print(f"Problem: Chaotic attractor (σ={sigma}, ρ={rho}, β={beta:.2f})")
    print(f"Time step: {dt}, Max time: {t_max}")
    print("=" * 60)
    
    # Step 1: Generate ground truth (for testing)
    print("\n[1/6] Generating ground truth...")
    t_true, y_true = generate_ground_truth(dt, t_max, sigma=sigma, rho=rho, beta=beta)
    print(f"Generated {len(t_true)} time points")
    
    # Step 2: Generate training data from MULTIPLE trajectories
    print("\n[2/6] Generating training dataset from multiple trajectories...")
    dataset = generate_training_data_multi_trajectory(dt, t_max, n_samples, sigma, rho, beta)
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
    y0 = np.array([1.0, 1.0, 1.0])  # Standard Lorenz initial condition
    n_steps = len(t_true) - 1
    
    # RK4 only
    y_rk4 = [y0.copy()]
    for i in range(n_steps):
        y_next = rk4_step(lorenz_rhs, i*dt, y_rk4[-1], dt, sigma, rho, beta)
        y_rk4.append(y_next)
    y_rk4 = np.array(y_rk4)
    
    # Hybrid (RK4 + Correction)
    y_hybrid = [y0.copy()]
    corr_net.eval()
    for i in range(n_steps):
        y_curr = y_hybrid[-1]
        t_curr = i * dt
        
        # RK4 step
        y_rk4_pred = rk4_step(lorenz_rhs, t_curr, y_curr, dt, sigma, rho, beta)
        
        # Neural correction
        with torch.no_grad():
            inp = torch.tensor([t_curr, y_curr[0], y_curr[1], y_curr[2], dt], dtype=torch.float32)
            correction = corr_net(inp).numpy()
            # Limit correction magnitude for stability
            correction = np.clip(correction, -2.0, 2.0)
        
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
    
    # Lyapunov time estimate (error doubling time)
    print(f"\nChaotic dynamics analysis:")
    print(f"  Initial error (t=1): RK4={error_rk4[int(1/dt)]:.6f}, Hybrid={error_hybrid[int(1/dt)]:.6f}")
    print(f"  Mid error (t={t_max/2:.0f}): RK4={error_rk4[len(error_rk4)//2]:.6f}, Hybrid={error_hybrid[len(error_hybrid)//2]:.6f}")
    
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

# Visualize Results - SEPARATE PLOTS
def plot_results(results):
    """Create 5 separate publication-quality plots for Lorenz system"""
    
    # Plot 1: Error Evolution Over Time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(results['t_true'], results['error_rk4'], 'r-', label='RK4', linewidth=2.5, alpha=0.8)
    ax1.plot(results['t_true'], results['error_hybrid'], 'g-', label='Hybrid (RK4 + NN)', linewidth=2.5, alpha=0.9)
    ax1.plot(results['t_true'], results['error_pinn'], 'orange', label='PINN', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('L2 Error', fontsize=14, fontweight='bold')
    ax1.set_title('Error Evolution Over Time - Lorenz System', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig('lorenz_error_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lorenz_error_plot.png")
    plt.show()
    
    # Plot 2: X component time series
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(results['t_true'], results['y_true'][:, 0], 'k--', 
             label='Ground Truth', linewidth=2.5, alpha=0.7, zorder=5)
    ax2.plot(results['t_true'], results['y_rk4'][:, 0], 'r-', 
             label='RK4', linewidth=2, alpha=0.6)
    ax2.plot(results['t_true'], results['y_hybrid'][:, 0], 'g-', 
             label='Hybrid (RK4 + NN)', linewidth=2, alpha=0.8)
    ax2.plot(results['t_true'], results['y_pinn'][:, 0], 'orange', 
             label='PINN', linewidth=2, alpha=0.6)
    ax2.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('x(t)', fontsize=14, fontweight='bold')
    ax2.set_title('X Component - Lorenz Attractor', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig('lorenz_x_component.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lorenz_x_component.png")
    plt.show()
    
    # Plot 3: 3D Butterfly Attractor - Ground Truth vs Hybrid
    print("Generating 3D plot (Hybrid)...")
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Downsample for faster rendering (every 5th point)
    skip = 5
    ax3.plot(results['y_true'][::skip, 0], results['y_true'][::skip, 1], results['y_true'][::skip, 2],
             'k-', label='Ground Truth', linewidth=1, alpha=0.5)
    
    # Plot hybrid
    ax3.plot(results['y_hybrid'][::skip, 0], results['y_hybrid'][::skip, 1], results['y_hybrid'][::skip, 2],
             'g-', label='Hybrid', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('X', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax3.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax3.set_title('3D Lorenz Attractor - Hybrid vs Truth', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10)
    ax3.view_init(elev=20, azim=45)
    plt.savefig('lorenz_3d_hybrid.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: lorenz_3d_hybrid.png")
    plt.close()
    
    # Plot 4: 3D Butterfly Attractor - RK4 vs Truth
    print("Generating 3D plot (RK4)...")
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    
    # Downsample for faster rendering
    ax4.plot(results['y_true'][::skip, 0], results['y_true'][::skip, 1], results['y_true'][::skip, 2],
             'k-', label='Ground Truth', linewidth=1, alpha=0.5)
    
    # Plot RK4
    ax4.plot(results['y_rk4'][::skip, 0], results['y_rk4'][::skip, 1], results['y_rk4'][::skip, 2],
             'r-', label='RK4', linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('X', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax4.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax4.set_title('3D Lorenz Attractor - RK4 vs Truth', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=10)
    ax4.view_init(elev=20, azim=45)
    plt.savefig('lorenz_3d_rk4.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: lorenz_3d_rk4.png")
    plt.close()
    
    # Plot 5: Phase Portrait (X-Z plane)
    print("Generating phase portrait...")
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    # Downsample for clarity
    skip = 3
    ax5.plot(results['y_true'][::skip, 0], results['y_true'][::skip, 2], 'k-', 
             label='Ground Truth', linewidth=1.5, alpha=0.5)
    ax5.plot(results['y_rk4'][::skip, 0], results['y_rk4'][::skip, 2], 'r-', 
             label='RK4', linewidth=1.5, alpha=0.7)
    ax5.plot(results['y_hybrid'][::skip, 0], results['y_hybrid'][::skip, 2], 'g-', 
             label='Hybrid', linewidth=1.5, alpha=0.9)
    ax5.set_xlabel('X', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Z', fontsize=14, fontweight='bold')
    ax5.set_title('Phase Portrait (X-Z Plane) - Lorenz Attractor', fontsize=16, fontweight='bold', pad=20)
    ax5.legend(fontsize=12, loc='best', framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig('lorenz_phase_portrait.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: lorenz_phase_portrait.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("ALL PLOTS SAVED SUCCESSFULLY!")
    print("=" * 60)
    print("Files created:")
    print("  1. lorenz_error_plot.png")
    print("  2. lorenz_x_component.png")
    print("  3. lorenz_3d_hybrid.png (Butterfly attractor!)")
    print("  4. lorenz_3d_rk4.png")
    print("  5. lorenz_phase_portrait.png")
    print("=" * 60)

# Run everything
if __name__ == "__main__":
    results = run_experiment(
        dt=0.02,          
        t_max=20.0,       
        n_samples=30000,
        sigma=10.0,       
        rho=28.0,
        beta=8.0/3.0
    )
    
    plot_results(results)