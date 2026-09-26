"""
CA/FA GAUSSIAN GENERATOR — KRÓTKA WERSJA
-----------------------------------------
- c_CA, c_FA: losowane ciągle z [1, 10]
- m(c), sigma(c), A(c): funkcje dopasowane na poziomach 3–10
- dla 1–2: ekstrapolacja
- zwykłe symetryczne Gaussiany
- szum JW 1% osobno dla CA i FA
- na końcu zwykła suma, bez żadnej poprawki
"""

import numpy as np


# Oś potencjału: 400–1102 mV co 2 mV
V = np.arange(400.0, 1102.0 + 2.0, 2.0)


# ---------- CA ----------
def m_CA(c):
    return 755.558334 + 0.308333 * c

def sigma_CA(c):
    return 85.573379 + 38.552369 * np.exp(-0.130403 * c)

def A_CA(c):
    return 0.03082171 + 0.01966242 * c


# ---------- FA ----------
def m_FA(c):
    return 767.321429 + 1.817857 * c

def sigma_FA(c):
    return 84.106205 + 66.957852 * np.exp(-0.564331 * c)

def A_FA(c):
    return 0.05497994 + 0.02470747 * c


def gaussian(A, m, sigma):
    return A * np.exp(-0.5 * ((V - m) / sigma) ** 2)


def jw_noise_1pct(y, A, rng):
    """Szum JW 1%: peak-to-peak szumu = 1% wysokości danego piku."""
    z = rng.normal(size=len(y))
    noise = 0.01 * A * z / (z.max() - z.min())
    return y + noise


def generate_one(rng):
    # Ciągłe, niezależne stężenia 1–10
    c_ca = rng.uniform(1.0, 10.0)
    c_fa = rng.uniform(1.0, 10.0)

    # Parametry Gaussianów
    p_ca = (A_CA(c_ca), m_CA(c_ca), sigma_CA(c_ca))
    p_fa = (A_FA(c_fa), m_FA(c_fa), sigma_FA(c_fa))

    # Dwa pojedyncze Gaussiany
    ca = gaussian(*p_ca)
    fa = gaussian(*p_fa)

    # Szum JW 1% osobno
    ca = jw_noise_1pct(ca, p_ca[0], rng)
    fa = jw_noise_1pct(fa, p_fa[0], rng)

    # Bez poprawki: zwykła suma
    signal = ca + fa

    # Etykiety do uczenia
    labels = [
        c_ca, c_fa,
        p_ca[0], p_ca[1], p_ca[2],
        p_fa[0], p_fa[1], p_fa[2],
    ]
    return signal, labels


def generate_dataset(n=1000, seed=20260917):
    rng = np.random.default_rng(seed)

    X = []
    Y = []

    for _ in range(n):
        signal, labels = generate_one(rng)
        X.append(signal)
        Y.append(labels)

    return np.asarray(X), np.asarray(Y)


if __name__ == "__main__":
    X, Y = generate_dataset(n=1000, seed=20260917)

    np.savez_compressed(
        "CA_FA_gaussian_dataset.npz",
        V_mV=V,
        X=X,
        Y=Y,
    )

    print("Gotowe.")
    print("X:", X.shape, "— zaszumione sumy CA+FA")
    print("Y:", Y.shape, "— [c_CA,c_FA,A_CA,m_CA,sigma_CA,A_FA,m_FA,sigma_FA]")
