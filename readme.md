# DEV NOTES

### TODO

- not sure if some $c(t-\tau)$ stuff is needed or not in the cross-moment differential equation
- it would be interesting to try an adaptive depth (i am not even sure it is possible to change the state size mid simulation, but hell we could try)

### 2025-08-19

Added `depth` to the solver.
This whole thing is similar to infinite determinants, more cross moments mean better precision.

**Testing on Mathieu**

- even `depth=1` was decent
- some deviation can be seen for `depth=10` on a $T=2\tau$ interval

### 2026-02-24

Nem tudtam rájönni, hogy mi a gond.
Az van, hogy az Sk diffegyenletek nem adják vissza szépen a `μ(t)*μ(t-kτ)'` szorzatokat determinisztikus esetben és így nem is jön ki, hogy az M az igazából μ*μ'.
Sokszor átnéztem a diffegyenletet, az jónak tűnik. Azt nem tudom, hogy esetleg a history-ban jól van-e figyelembe véve, pl. Sk1-et még τ-ig nem kellene integrálni, vagy bánat tudja.
Tesztelgetek csak sima determinisztikus rendszereket és ha `Sk = μ(t)*μ(t-kτ)` akkor visszaadja a második momentum az első négyzetét, de az szerintem nem lesz jó a sztochasztikus esetekben. Vagy nem tudom, lehet, hogy érdemes lenne kipróbálni.
