import Mathlib.Tactic

def tokenSavings (n r T : ℕ) (h : r ≤ n) : ℕ := n * T - r * T

lemma token_savings_expansion (n r T : ℕ) (h : r ≤ n) :
    tokenSavings n r T h = (n - r) * T :=
  (Nat.mul_sub_right_distrib n r T).symm

lemma savings_monotone_in_rank (n r₁ r₂ T : ℕ) (h1 : r₁ ≤ r₂) (h2 : r₂ ≤ n)
    (hr1 : r₁ ≤ n) :
    tokenSavings n r₂ T h2 ≤ tokenSavings n r₁ T hr1 := by
  rw [token_savings_expansion n r₂ T h2, token_savings_expansion n r₁ T hr1]
  have : n - r₂ ≤ n - r₁ := Nat.sub_le_sub_left h1 n
  exact Nat.mul_le_mul_right T this

theorem savings_maximal_at_rank_zero (n T : ℕ) (r : ℕ) (hr : r ≤ n) :
    tokenSavings n r T hr ≤ tokenSavings n 0 T (Nat.zero_le n) := by
  apply savings_monotone_in_rank
  · exact Nat.zero_le r
