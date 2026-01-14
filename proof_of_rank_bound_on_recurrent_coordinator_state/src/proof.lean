import Mathlib.Tactic

-- Parameter count for a rank-r factorization of a d×d matrix
-- A rank-r factorization requires two matrices: U (d×r) and V (r×d)
-- Total parameters: d*r + r*d = 2*r*d
def lowRankParamCount (d r : ℕ) : ℕ := 2 * r * d

-- Parameter count for a full-rank d×d matrix
def fullRankParamCount (d : ℕ) : ℕ := d * d

-- Lemma 1: Basic arithmetic property
lemma two_rd_eq (r d : ℕ) : 2 * r * d = r * d + r * d := by
  ring

-- Lemma 2: Low-rank has O(r*d) parameters
lemma param_is_O_rd (d r : ℕ) : lowRankParamCount d r = 2 * r * d := by
  rfl

-- Lemma 3: When 2*r < d, low-rank uses fewer parameters than full-rank
lemma lowRank_fewer_params (d r : ℕ) (h : 2 * r < d) (hd : 0 < d) :
  lowRankParamCount d r < fullRankParamCount d := by
  unfold lowRankParamCount fullRankParamCount
  have h1 : 2 * r * d < d * d := by
    calc 2 * r * d
        = (2 * r) * d := by ring
      _ < d * d := by
          apply Nat.mul_lt_mul_of_lt_of_le h (Nat.le_refl d) hd
  exact h1

-- Lemma 4: For typical case r ≤ d/2, the bound holds
lemma bound_holds_typical (d r : ℕ) (hr : 0 < r) (hbound : 2 * r ≤ d) :
  lowRankParamCount d r ≤ d * d := by
  unfold lowRankParamCount
  cases' d with d'
  · omega
  · have : 2 * r ≤ Nat.succ d' := hbound
    calc 2 * r * Nat.succ d'
        ≤ Nat.succ d' * Nat.succ d' := by
          apply Nat.mul_le_mul_right
          exact this

-- Main Theorem: Rank-bound on recurrent coordinator state
-- Proves that a rank-r coordinator uses O(r·d) parameters and is more efficient
theorem recurrent_coordinator_rank_bound (d r : ℕ) (hr : 0 < r) (hrd : r < d) (hd : 1 < d) :
  ∃ (paramCount : ℕ),
    paramCount = lowRankParamCount d r ∧
    paramCount = 2 * r * d ∧
    (∃ (c : ℕ), paramCount ≤ c * r * d) ∧
    (2 * r < d → paramCount < fullRankParamCount d) := by
  use lowRankParamCount d r
  constructor
  · rfl
  constructor
  · exact param_is_O_rd d r
  constructor
  · use 2
    exact Nat.le_refl _
  · intro h_2r_lt_d
    exact lowRank_fewer_params d r h_2r_lt_d (Nat.zero_lt_of_lt hd)
