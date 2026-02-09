# Hierarchical RL Training Strategies Comparison

## 1. SEQUENTIAL (Recommended)

**Procedure:**
1. Train micro-RL agents on fixed "average" infrastructure (500 episodes)
2. Freeze micro-RL policies
3. Train macro-RL using frozen micro-agents (300 episodes)

**Pros:**
- ✅ Stable training
- ✅ Clear credit assignment
- ✅ Micro-agents learn optimal arbitrage before infrastructure changes
- ✅ 2-3x faster total training time
- ✅ Better final performance

**Cons:**
- ❌ Requires two training phases
- ❌ Micro-agents may not generalize to all infrastructure configs

**Best For:** Production use, reliable results

**Expected Time:** 4-6 hours for 3 stations

---

## 2. SIMULTANEOUS

**Procedure:**
- Train both macro and micro at same time
- Micro updates every hour, macro updates every 10 episodes

**Pros:**
- ✅ Single training phase
- ✅ Micro adapts to specific infrastructure immediately
- ✅ Potential for emergent co-adaptation

**Cons:**
- ❌ Highly unstable
- ❌ Non-stationary problem for both agents
- ❌ Requires careful tuning
- ❌ Often diverges

**Best For:** Research, exploring agent interactions

**Expected Time:** 8-12 hours (with restarts)

---

## 3. CURRICULUM (Most Robust)

**Procedure:**
- Stage 1: Train micro with fixed prices (simple)
- Stage 2: Train macro with simple micro (1-day episodes)
- Stage 3: Fine-tune with variable prices (7-day episodes)

**Pros:**
- ✅ Most robust convergence
- ✅ Best generalization
- ✅ Handles complexity gradually

**Cons:**
- ❌ Longest training time
- ❌ More hyperparameters

**Best For:** Complex scenarios, publication-quality results

**Expected Time:** 12-24 hours

---

## Recommendation

**For most users: Use SEQUENTIAL mode**

```python
from HierarchicalTrainer import train_efficient

result = train_efficient(
    n_stations=3,
    highway_length=300.0,
    output_dir="./my_models"
)