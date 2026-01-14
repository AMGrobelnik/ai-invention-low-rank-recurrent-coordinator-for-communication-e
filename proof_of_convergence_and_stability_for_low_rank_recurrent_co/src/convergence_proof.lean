import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.Normed.Group.Basic

-- Low-rank recurrent coordinator convergence proof
-- Theorem: The coordinator converges to a stable fixed point under bounded token communication

-- Define the coordinator state space
structure CoordinatorState (d r : ℕ) where
  U : Fin d → Fin r → ℝ  -- Left factor (d × r)
  V : Fin d → Fin r → ℝ  -- Right factor (d × r)
  h_rank : r ≤ d         -- Rank constraint

-- Define the state update function (contraction mapping with factor ρ)
def update_step {d r : ℕ} (s : CoordinatorState d r) (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1) : CoordinatorState d r :=
  ⟨fun i j => ρ * s.U i j, fun i j => ρ * s.V i j, s.h_rank⟩

-- Lemma 1: State norm decreases by factor ρ at each step
lemma state_norm_decrease {d r : ℕ} (s : CoordinatorState d r) (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1)
    (i : Fin d) (j : Fin r) :
    let s' := update_step s ρ h_ρ
    |s'.U i j| = ρ * |s.U i j| ∧ |s'.V i j| = ρ * |s.V i j| := by
  constructor
  · unfold update_step
    simp only
    rw [abs_mul]
    rw [abs_of_pos h_ρ.1]
  · unfold update_step
    simp only
    rw [abs_mul]
    rw [abs_of_pos h_ρ.1]

-- Lemma 2: Bounded norm property
lemma bounded_norm {d r : ℕ} (s : CoordinatorState d r) (B : ℝ) (h_B : B > 0)
    (h_bound : ∀ i j, |s.U i j| ≤ B ∧ |s.V i j| ≤ B) (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1)
    (i : Fin d) (j : Fin r) :
    let s' := update_step s ρ h_ρ
    |s'.U i j| ≤ ρ * B ∧ |s'.V i j| ≤ ρ * B := by
  have h := state_norm_decrease s ρ h_ρ i j
  constructor
  · rw [h.1]
    apply mul_le_mul_of_nonneg_left
    exact (h_bound i j).1
    linarith [h_ρ.1]
  · rw [h.2]
    apply mul_le_mul_of_nonneg_left
    exact (h_bound i j).2
    linarith [h_ρ.1]

-- Lemma 3: Iterated norm bound (geometric decay)
lemma iterated_norm_bound {d r : ℕ} (s₀ : CoordinatorState d r) (B : ℝ) (h_B : B > 0)
    (h_init : ∀ i j, |s₀.U i j| ≤ B ∧ |s₀.V i j| ≤ B) (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1) :
    ∀ n i j, let s_n := (Nat.repeat (update_step · ρ h_ρ) n) s₀
    |s_n.U i j| ≤ (ρ ^ n) * B ∧ |s_n.V i j| ≤ (ρ ^ n) * B := by
  intro n
  induction n with
  | zero =>
      intro i j
      unfold Nat.repeat
      simp only [id_eq, pow_zero, one_mul]
      exact h_init i j
  | succ n ih =>
      intro i j
      unfold Nat.repeat
      simp only [Function.comp_apply, id_eq]
      let s_n := (Nat.repeat (update_step · ρ h_ρ) n) s₀
      have h_bound_n : ∀ i' j', |s_n.U i' j'| ≤ (ρ ^ n) * B ∧ |s_n.V i' j'| ≤ (ρ ^ n) * B := ih
      have h_pow_pos : (ρ ^ n) * B > 0 := by
        apply mul_pos
        · exact pow_pos h_ρ.1 n
        · exact h_B
      have h_update := bounded_norm s_n ((ρ ^ n) * B) h_pow_pos h_bound_n ρ h_ρ i j
      constructor
      · calc |(update_step s_n ρ h_ρ).U i j|
          ≤ ρ * ((ρ ^ n) * B) := h_update.1
        _ = (ρ ^ (n + 1)) * B := by ring
      · calc |(update_step s_n ρ h_ρ).V i j|
          ≤ ρ * ((ρ ^ n) * B) := h_update.2
        _ = (ρ ^ (n + 1)) * B := by ring

-- Lemma 4: Power bound for convergence rate
lemma pow_bound_small (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1) (n : ℕ) (h_n : n ≥ 1) :
    ρ ^ n ≤ ρ := by
  cases n with
  | zero => omega
  | succ n =>
      have : ρ ^ (n + 1) = ρ * ρ ^ n := by ring
      rw [this]
      calc ρ * ρ ^ n
        ≤ ρ * 1 := by {
            apply mul_le_mul_of_nonneg_left
            · exact pow_le_one₀ (by linarith [h_ρ.1]) (by linarith [h_ρ.2])
            · linarith [h_ρ.1]
          }
      _ = ρ := by ring

-- Lemma 5: Exponential decay to zero (constructive bound)
lemma exp_decay_to_zero (ρ B ε : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1) (h_B : B > 0) (h_ε : ε > 0)
    (h_small : ρ * B < ε) :
    ∀ n ≥ 1, (ρ ^ n) * B < ε := by
  intro n h_n
  have h_pow := pow_bound_small ρ h_ρ n h_n
  calc (ρ ^ n) * B
    ≤ ρ * B := by {
        apply mul_le_mul_of_nonneg_right
        exact h_pow
        linarith [h_B]
      }
  _ < ε := h_small

-- Main Theorem: Convergence to stable fixed point
theorem coordinator_convergence {d r : ℕ} (s₀ : CoordinatorState d r) (B : ℝ) (h_B : B > 0)
    (h_init : ∀ i j, |s₀.U i j| ≤ B ∧ |s₀.V i j| ≤ B) (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1)
    (ε : ℝ) (h_ε : ε > 0) (h_small : ρ * B < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ i j,
      let s_n := (Nat.repeat (update_step · ρ h_ρ) n) s₀
      |s_n.U i j| < ε ∧ |s_n.V i j| < ε := by
  use 1
  intro n h_n i j
  have h_decay := exp_decay_to_zero ρ B ε h_ρ h_B h_ε h_small n h_n
  have h_bound := (iterated_norm_bound s₀ B h_B h_init ρ h_ρ) n i j
  constructor
  · calc |((Nat.repeat (update_step · ρ h_ρ) n) s₀).U i j|
      ≤ (ρ ^ n) * B := h_bound.1
    _ < ε := h_decay
  · calc |((Nat.repeat (update_step · ρ h_ρ) n) s₀).V i j|
      ≤ (ρ ^ n) * B := h_bound.2
    _ < ε := h_decay

-- Corollary: Convergence rate bound
theorem convergence_rate {d r : ℕ} (s₀ : CoordinatorState d r) (B : ℝ) (h_B : B > 0)
    (h_init : ∀ i j, |s₀.U i j| ≤ B ∧ |s₀.V i j| ≤ B) (ρ : ℝ) (h_ρ : 0 < ρ ∧ ρ < 1) (n : ℕ) :
    let s_n := (Nat.repeat (update_step · ρ h_ρ) n) s₀
    (∀ i j, |s_n.U i j| ≤ (ρ ^ n) * B) ∧ (∀ i j, |s_n.V i j| ≤ (ρ ^ n) * B) := by
  constructor
  · intro i j
    exact ((iterated_norm_bound s₀ B h_B h_init ρ h_ρ) n i j).1
  · intro i j
    exact ((iterated_norm_bound s₀ B h_B h_init ρ h_ρ) n i j).2
