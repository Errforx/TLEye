# RL Agent Training Guide

## 📊 How the Agent Learns

The DQN agent uses **Q-learning** to optimize alert decisions based on:
- **State**: brightness, detection confidence, bbox area, tracking age, variance, time since last alert
- **Actions**: 
  - 0 = do nothing
  - 1-8 = tune parameters (gamma, confidence, IoU, buffer)
  - 9 = trigger alert
  - 10 = suppress alert
- **Reward**: Strong signal when actions lead to correct/incorrect alerts

## 🧪 Training Phases

### Phase 1: Initial Exploration (Hours 1-4)
- Agent explores randomly (ε=1.0 initially, decays to 0.05)
- Experiences reward signals
- Builds replay buffer
- **Expect**: Many false alerts initially—this is normal!

### Phase 2: Convergence (Hours 4-12)
- Epsilon decays, agent exploits learned policy
- Should see fewer false alerts
- Parameters start stabilizing
- **Monitor**: Check logs for reward trend

### Phase 3: Refinement (Day 2+)
- Agent fine-tunes alert timing
- Should only alert on real emergencies
- Adapts to lighting/noise patterns

## 📈 Monitoring Training

Check these metrics:

```python
# View during running:
# - Average reward per 100 frames
# - Actions chosen (should see more 0, 9, 10 over time)
# - Alert accuracy (true positive rate)
```

## ⚙️ Hyperparameters You Can Tune

Open `rl_agent.py` and adjust:

```python
RLAgent(
    STATE_SIZE=6,
    ACTION_SIZE=11,
    lr=1e-3,              # Learning rate (0.5e-3 to 5e-3)
    gamma=0.99,           # Discount factor (0.95-0.99)
    epsilon_start=1.0,    # Initial exploration
    epsilon_end=0.05,     # Final exploitation
    epsilon_decay=0.995,  # How fast to decay (0.99-0.999)
    buffer_size=10000,    # Replay memory (5000-50000)
    batch_size=64,        # Training batch (32-128)
)
```

## 🎯 Expected Results

After good training:
- **False alerts**: Near 0
- **Missed alerts**: Near 0  
- **Parameter adaptation**: Gamma/confidence adjust to scene conditions
- **Peak accuracy**: 95%+ true alert rate after 24-48 hours

## 💾 Persistence

Model automatically saves to `rl_model.pth` on shutdown. Resume training anytime:

```python
# The agent auto-loads last weights at startup
if os.path.exists("rl_model.pth"):
    rl_agent.load("rl_model.pth")  # ✅ Training continues
```

## 🚨 Safety Notes

- **First run**: Monitor alerts closely—agent may fire randomly
- **Strong negative reward** for false alerts guides learning quickly
- After 4-6 hours: False alert rate should drop significantly
- After 24 hours: Should approach near-perfect accuracy

## 🔄 Reset Training (if needed)

Delete `rl_model.pth` to start fresh:
```bash
rm rl_model.pth
python app.py  # New agent from scratch
```

---

**Key Insight**: The **reward function is the curriculum**. Strong penalties for false/missed alerts teach the agent conservative, accurate decision-making.
