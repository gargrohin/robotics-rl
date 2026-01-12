# Blog: The Reward Shaping Trap in Robotics RL

## Hook
- Open with the "grasp and hold forever" example - a robot that achieves 250 reward by holding the cube but refuses to lift it, while successful episodes only score 5
- The gap between what seems like a reasonable reward and what the agent actually optimizes for

## What is Reward Shaping?
- Definition: adding intermediate rewards to guide learning toward sparse goals
- Why it's necessary: sparse rewards (success/fail) are hard to learn from
- The trap: shaped rewards can create unintended incentives
- Dense rewards feel helpful but can backfire spectacularly

## The Experiment Setup

**Task:** Robotic manipulation - pick up a cube and lift it 4cm (robosuite Lift environment)

**Algorithm:** PPO with continuous action space (7D: arm + gripper)

**Initial Reward (robosuite default):**
- Reaching: ~0.4/step based on gripper-to-cube distance
- Grasp bonus: 0.25 (one-time)
- Success bonus: 2.25 (one-time)

**Problem:** Reaching reward dominates. Agent learns to hover near cube forever (200+ reward) without ever grasping (2.5 reward).

## Iteration 1: Rebalancing Rewards

**Changes:**
- Reduced reaching weight: 0.1
- Increased grasp reward: 0.5/step (continuous)
- Added lift reward: 1.0/step (scaled by height)
- Increased success: 5.0

**Result:** 40% success rate! But...

**New problem observed:**
- Successful episodes: ~5 cumulative reward
- Failed episodes (grasp + hold): ~250 cumulative reward
- Agent incentivized to hold forever, not complete the task

## Iteration 2: Time Penalty

**Intuition:** Penalize each timestep so holding forever isn't free

**Changes:**
- Added time_penalty: 0.5/step
- Increased success_reward: 100

**Expected:** Grasp reward (0.5) cancels time penalty (0.5), success bonus makes completion best option

**Result:** Agent stuck at -250 reward. Never grasps at all.

**What went wrong:** Time penalty punishes exploration. Every step costs 0.5, and the reaching reward is too weak to guide the agent toward the cube.

## Iteration 3: The Gradient Problem

**Discovery:** The reaching reward formula was the hidden culprit

```python
dist_reward = 1 - tanh(10 * dist)
```

The `tanh(10*dist)` saturates quickly:
- dist > 0.3m → reward ≈ 0 (essentially sparse!)
- dist < 0.1m → reward ≈ 1

**The agent only "felt" the reaching reward when already very close** - but random exploration rarely gets that close.

**Fix:** Softer coefficient for gentler gradient
```python
dist_reward = 1 - tanh(3 * dist)  # or even 2
```

## Key Lessons

**1. Cumulative rewards matter more than per-step rewards**
- A small per-step reward × 500 steps = massive total
- Always calculate episode totals for different behaviors

**2. Reward gradients need to be learnable**
- Sharp transitions (like tanh with high coefficient) create sparse regions
- The agent needs a gradient it can follow from any starting state

**3. Time pressure is tricky**
- Time penalty can kill exploration if too high
- Must balance against other reward components
- Alternative: make success reward so large that speed doesn't matter

**4. Watch your trained policy, not just the reward curve**
- High reward ≠ desired behavior
- The "grasp and hold" policy had excellent reward but useless behavior

**5. Visualize your reward landscape**
- Plot reward vs state variables before training
- Calculate episode outcomes for different behaviors
- [Link to reward visualization notebook]

## The Reward Designer's Checklist

Before training, verify:
- [ ] Success gives highest episode reward (for any reasonable episode length)
- [ ] Reward gradient exists from initial state to goal
- [ ] Intermediate rewards don't dominate terminal rewards
- [ ] Time penalty (if any) doesn't overwhelm exploration signal
- [ ] Multiple behaviors calculated: random, partial progress, success

## Broader Implications

- Reward shaping is necessary but dangerous
- "Reward hacking" happens even with dense, shaped rewards
- The optimizer will find whatever you're actually rewarding
- Iteration and observation are unavoidable - expect to tune

## Takeaway

- Your intuition about "good" rewards will be wrong
- Always simulate episode rewards before training
- Watch the trained policy, not just the metrics
- Reward design is empirical - build tools to visualize and iterate quickly
