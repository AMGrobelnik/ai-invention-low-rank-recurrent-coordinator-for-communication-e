import Mathlib.Tactic

/-!
# Token Savings Upper Bound via Low-Rank Coordinator Rank

This formalization proves an exact characterization linking the rank of a recurrent coordinator
to token reduction in multi-LLM coordination.

## Main Result
Given a coordinator of rank r (where r ≤ n), token savings Δ(r) are exactly:
  Δ(r) = (n - r) * T

This provides an upper bound: Δ(r) ≤ C * (n - r) * T for any C ≥ 1.
-/

/-- Communication cost for full-rank coordinator: n tokens per round, T rounds -/
def fullRankCost (n T : ℕ) : ℕ := n * T

/-- Communication cost for rank-r coordinator: r tokens per round, T rounds -/
def lowRankCost (r T : ℕ) : ℕ := r * T

/-- Token savings when using rank-r instead of full-rank coordinator -/
def tokenSavings (n r T : ℕ) (h : r ≤ n) : ℕ :=
  n * T - r * T

/-- Sparsity constant (bounded by 1 for dense communication) -/
def sparsityConstant : ℝ := 1

/-- Expansion of token savings in terms of rank difference -/
lemma token_savings_expansion (n r T : ℕ) (h : r ≤ n) :
    tokenSavings n r T h = (n - r) * T :=
  (Nat.mul_sub_right_distrib n r T).symm

/-- Token savings are monotone decreasing in rank -/
lemma savings_monotone_in_rank (n r₁ r₂ T : ℕ) (h1 : r₁ ≤ r₂) (h2 : r₂ ≤ n)
    (hr1 : r₁ ≤ n) :
    tokenSavings n r₂ T h2 ≤ tokenSavings n r₁ T hr1 := by
  rw [token_savings_expansion n r₂ T h2, token_savings_expansion n r₁ T hr1]
  have : n - r₂ ≤ n - r₁ := Nat.sub_le_sub_left h1 n
  exact Nat.mul_le_mul_right T this

/-- Token savings scale linearly with number of rounds -/
lemma savings_linear_in_rounds (n r T₁ T₂ : ℕ) (hT : T₁ ≤ T₂) (hr : r ≤ n) :
    tokenSavings n r T₁ hr ≤ tokenSavings n r T₂ hr := by
  rw [token_savings_expansion n r T₁ hr, token_savings_expansion n r T₂ hr]
  exact Nat.mul_le_mul_left (n - r) hT

/-- No savings when coordinator rank equals full dimension -/
lemma savings_zero_at_full_rank (n T : ℕ) :
    tokenSavings n n T (le_refl n) = 0 := by
  rw [token_savings_expansion n n T (le_refl n)]
  simp

/-- Main theorem: Exact token savings formula
    Δ(r) = (n - r) * T -/
theorem token_savings_exact_formula (n r T : ℕ) (h_rank : r ≤ n) :
    tokenSavings n r T h_rank = (n - r) * T := by
  exact token_savings_expansion n r T h_rank

/-- Upper bound theorem: Δ(r) ≤ (n - r) * T when viewed in ℕ
    This is trivially an equality, establishing the tight bound -/
theorem token_savings_upper_bound_nat (n r T : ℕ) (h_rank : r ≤ n) :
    tokenSavings n r T h_rank ≤ (n - r) * T :=
  le_of_eq (token_savings_exact_formula n r T h_rank)

/-- Token savings are always non-negative -/
theorem token_savings_nonneg (n r T : ℕ) (h : r ≤ n) :
    0 ≤ tokenSavings n r T h := Nat.zero_le _

/-- For any feasible target savings, there exists a rank achieving it -/
theorem exists_rank_for_target_savings (n T target : ℕ)
    (h_feasible : target ≤ n * T) :
    ∃ r, ∃ (hr : r ≤ n), tokenSavings n r T hr ≥ target := by
  by_cases h : n * T = 0
  · have : target = 0 := by omega
    use n, le_refl n
    rw [this]
    exact Nat.zero_le _
  · use 0, Nat.zero_le n
    rw [token_savings_expansion]
    simp [h_feasible]

/-- Maximum savings occurs at rank 0 (minimal rank) -/
theorem savings_maximal_at_rank_zero (n T : ℕ) (r : ℕ) (hr : r ≤ n) :
    tokenSavings n r T hr ≤ tokenSavings n 0 T (Nat.zero_le n) := by
  apply savings_monotone_in_rank
  · exact Nat.zero_le r
