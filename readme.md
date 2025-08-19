# DEV NOTES

### TODO

- not sure if some $c(t-\tau)$ stuff is needed or not in the cross-moment differential equation

### 2025-08-19

Added `depth` to the solver.
This whole thing is similar to infinite determinants, more cross moments mean better precision.

**Testing on Mathieu**

- even `depth=1` was decent
- some deviation can be seen for `depth=10` on a $T=2\tau$ interval
