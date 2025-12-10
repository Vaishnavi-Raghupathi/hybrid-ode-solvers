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

# Define Correction Neural Network
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

# ============================================================================
# TRUE PHYSICS-INFORMED NEURAL NETWORK (PINN) FOR ROBERTSON PROBLEM
# ============================================================================

class RobertsonPINN(nn.Module):
    """
    Physics-Informed Neural Network for Robertson problem
    Network learns y(t) while respecting the Robertson chemistry equations
    """
    def __init__(self, hidden_size=128):
        super().__init__()
        
        # Neural network: t -> [y1, y2, y3]
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3)  # Output: [y1, y2, y3]
        )
    
    def forward(self, t):
        """Forward pass through network"""
        # Apply softmax-like scaling to encourage conservation
        y = self.net(t)
        # Use sigmoid to keep values between 0 and 1
        y = torch.sigmoid(y)
        return y


def compute_robertson_physics_loss(model, t_physics):
    """
    Compute how well the network satisfies the Robertson ODE system
    
    Robertson equations:
    - dy1/dt = -0.04*y1 + 1e4*y2*y3
    - dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    - dy3/dt = 3e7*y2^2
    
    We penalize deviations from these equations!
    """
    # Enable gradient computation for input
    t_physics.requires_grad = True
    
    # Get network prediction
    y_pred = model(t_physics)
    y1 = y_pred[:, 0:1]
    y2 = y_pred[:, 1:2]
    y3 = y_pred[:, 2:3]
    
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
        create_graph=True,
        retain_graph=True
    )[0]
    
    dy3_dt = torch.autograd.grad(
        outputs=y3,
        inputs=t_physics,
        grad_outputs=torch.ones_like(y3),
        create_graph=True
    )[0]
    
    # Physics residuals (how much we violate the Robertson ODEs)
    # Equation 1: dy1/dt + 0.04*y1 - 1e4*y2*y3 = 0
    physics_residual_1 = dy1_dt + 0.04 * y1 - 1e4 * y2 * y3
    
    # Equation 2: dy2/dt - 0.04*y1 + 1e4*y2*y3 + 3e7*y2^2 = 0
    physics_residual_2 = dy2_dt - 0.04 * y1 + 1e4 * y2 * y3 + 3e7 * y2**2
    
    # Equation 3: dy3/dt - 3e7*y2^2 = 0
    physics_residual_3 = dy3_dt - 3e7 * y2**2
    
    # We want these residuals to be ZERO (perfect physics satisfaction)
    physics_loss = (torch.mean(physics_residual_1**2) + 
                   torch.mean(physics_residual_2**2) + 
                   torch.mean(physics_residual_3**2))
    
    return physics_loss


def compute_robertson_conservation_loss(model, t_physics):
    """
    Conservation law: y1 + y2 + y3 = 1 (always!)
    This is critical for Robertson problem
    """
    y_pred = model(t_physics)
    conservation = torch.sum(y_pred, dim=1, keepdim=True)
    conservation_loss = torch.mean((conservation - 1.0)**2)
    return conservation_loss


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


def train_robertson_pinn(t_data, y_data, epochs=3000, lr=0.001,
                         lambda_data=1.0, lambda_physics=0.001, 
                         lambda_ic=100.0, lambda_conservation=10.0):
    """
    Train PINN for Robertson problem with four loss components:
    1. Data loss: Fit the observed data
    2. Physics loss: Satisfy the Robertson ODEs
    3. Initial condition loss: Match initial conditions
    4. Conservation loss: Enforce y1 + y2 + y3 = 1
    """
    model = RobertsonPINN(hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)
    
    # Convert data to tensors
    t_data_tensor = torch.tensor(t_data.reshape(-1, 1), dtype=torch.float32)
    y_data_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Initial conditions
    t0 = torch.tensor([[0.0]], dtype=torch.float32)
    y0 = torch.tensor([[y_data[0, 0], y_data[0, 1], y_data[0, 2]]], dtype=torch.float32)
    
    # Physics collocation points (logarithmically spaced for stiff problems)
    n_physics = 2000
    t_physics_np = np.logspace(-6, np.log10(t_data[-1]), n_physics)
    t_physics = torch.tensor(t_physics_np.reshape(-1, 1), dtype=torch.float32)
    
    losses_total = []
    losses_data = []
    losses_physics = []
    losses_ic = []
    losses_conservation = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute all loss components
        loss_data = compute_pinn_data_loss(model, t_data_tensor, y_data_tensor)
        loss_physics = compute_robertson_physics_loss(model, t_physics)
        loss_ic = compute_pinn_ic_loss(model, t0, y0)
        loss_conservation = compute_robertson_conservation_loss(model, t_physics)
        
        # Weighted combination
        total_loss = (lambda_data * loss_data + 
                     lambda_physics * loss_physics + 
                     lambda_ic * loss_ic +
                     lambda_conservation * loss_conservation)
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record losses
        losses_total.append(total_loss.item())
        losses_data.append(loss_data.item())
        losses_physics.append(loss_physics.item())
        losses_ic.append(loss_ic.item())
        losses_conservation.append(loss_conservation.item())
        
        scheduler.step(total_loss)
        
        if epoch % 300 == 0:
            print(f"PINN Epoch {epoch:4d}/{epochs} | Total: {total_loss.item():.6f} | "
                  f"Data: {loss_data.item():.6f} | Physics: {loss_physics.item():.2e} | "
                  f"IC: {loss_ic.item():.2e} | Conservation: {loss_conservation.item():.2e}")
    
    return model, (losses_total, losses_data, losses_physics, losses_ic, losses_conservation)


# Run Full Experiment
def run_experiment(dt=0.01, t_max=10.0, n_samples=30000):
    """Complete experimental pipeline for Robertson problem"""
    
    print("=" * 70)
    print("HYBRID ODE SOLVER - ROBERTSON STIFF PROBLEM WITH TRUE PINN")
    print("=" * 70)
    print(f"Problem: Stiff chemical kinetics (3 species)")
    print(f"Time step: {dt}, Max time: {t_max}")
    print("=" * 70)
    
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
    
    # Step 4: Train TRUE PINN
    print("\n[4/6] Training TRUE PINN (Physics-Informed)...")
    # Use sparse data for PINN (every 20th point)
    t_pinn_data = t_true[::20]
    y_pinn_data = y_true[::20]
    print(f"PINN training with {len(t_pinn_data)} sparse data points")
    pinn, pinn_losses = train_robertson_pinn(
        t_pinn_data, y_pinn_data, epochs=3000, lr=0.001,
        lambda_data=1.0, lambda_physics=0.001, 
        lambda_ic=100.0, lambda_conservation=10.0
    )
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
    
    # TRUE PINN - direct prediction
    pinn.eval()
    with torch.no_grad():
        t_pinn = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32)
        y_pinn = pinn(t_pinn).numpy()
        # Already constrained by sigmoid in network, but clip for safety
        y_pinn = np.clip(y_pinn, 0, 1)
    
    # Step 6: Calculate errors
    print("\n[6/6] Calculating errors...")
    error_rk4 = np.linalg.norm(y_rk4 - y_true, axis=1)
    error_hybrid = np.linalg.norm(y_hybrid - y_true, axis=1)
    error_pinn = np.linalg.norm(y_pinn - y_true, axis=1)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
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
    
    # Conservation check
    print(f"\nConservation (y1+y2+y3 at t={t_max}):")
    print(f"  Truth:       {y_true[-1].sum():.8f}")
    print(f"  RK4:         {y_rk4[-1].sum():.8f}")
    print(f"  Hybrid:      {y_hybrid[-1].sum():.8f}")
    print(f"  TRUE PINN:   {y_pinn[-1].sum():.8f}")
    
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
    """Create publication-quality plots for Robertson problem"""
    fig = plt.figure(figsize=(16, 12))
    
    # Create 4x2 grid
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Error vs Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results['t_true'], results['error_rk4'], 'r-', label='RK4', linewidth=2, alpha=0.8)
    ax1.plot(results['t_true'], results['error_hybrid'], 'g-', label='Hybrid (RK4 + NN)', linewidth=2, alpha=0.8)
    ax1.plot(results['t_true'], results['error_pinn'], 'b-', label='TRUE PINN', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Error Evolution Over Time - Robertson Problem', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot 2-4: Individual species trajectories
    species_names = ['y₁ (Species A)', 'y₂ (Species B)', 'y₃ (Species C)']
    
    for idx, species_name in enumerate(species_names):
        ax = fig.add_subplot(gs[1 + idx//2, idx%2])
        
        ax.plot(results['t_true'], results['y_true'][:, idx], 'k--', 
                label='Ground Truth', linewidth=3, alpha=0.7)
        ax.plot(results['t_true'], results['y_rk4'][:, idx], 'r-', 
                label='RK4', linewidth=2, alpha=0.6)
        ax.plot(results['t_true'], results['y_hybrid'][:, idx], 'g-', 
                label='Hybrid', linewidth=2, alpha=0.8)
        ax.plot(results['t_true'], results['y_pinn'][:, idx], 'b-', 
                label='TRUE PINN', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Concentration', fontsize=11)
        ax.set_title(f'{species_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Plot 5: PINN Training Losses
    ax5 = fig.add_subplot(gs[2:, 1])
    losses_total, losses_data, losses_physics, losses_ic, losses_conservation = results['pinn_losses']
    epochs = np.arange(len(losses_total))
    
    ax5.plot(epochs, losses_total, 'k-', label='Total Loss', linewidth=2)
    ax5.plot(epochs, losses_data, 'b-', label='Data Loss', linewidth=1.5, alpha=0.7)
    ax5.plot(epochs, losses_physics, 'r-', label='Physics Loss', linewidth=1.5, alpha=0.7)
    ax5.plot(epochs, losses_ic, 'g-', label='IC Loss', linewidth=1.5, alpha=0.7)
    ax5.plot(epochs, losses_conservation, 'm-', label='Conservation Loss', linewidth=1.5, alpha=0.7)
    
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('PINN Training: Loss Components', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    plt.savefig('robertson_true_pinn_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved figure as 'robertson_true_pinn_results.png'")
    plt.show()

# Run everything
if __name__ == "__main__":
    results = run_experiment(
        dt=0.05,          
        t_max=10.0,       
        n_samples=30000   
    )
    
    plot_results(results)