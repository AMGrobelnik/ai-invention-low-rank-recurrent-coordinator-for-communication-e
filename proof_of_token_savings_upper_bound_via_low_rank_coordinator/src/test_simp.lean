import Mathlib.Tactic

def tokenSavings (n r T : ℕ) (h : r ≤ n) : ℕ := n * T - r * T

lemma token_savings_expansion (n r T : ℕ) (h : r ≤ n) :
    tokenSavings n r T h = (n - r) * T :=
  (Nat.mul_sub_right_distrib n r T).symm

theorem test (n T target : ℕ) (h_feasible : target ≤ n * T) (h : ¬n * T = 0) :
    ∃ r, ∃ (hr : r ≤ n), tokenSavings n r T hr ≥ target := by
  use 0, Nat.zero_le n
  rw [token_savings_expansion]
  simp [h_feasible]
