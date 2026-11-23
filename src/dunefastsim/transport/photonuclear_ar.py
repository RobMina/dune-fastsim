"""Photonuclear cross section of argon (Z = 18) as Geant4 11.4 option 0 + G4EmExtraPhysics computes it.

The reference (edep-sim, QGSP_BERT) adds G4EmExtraPhysics ("G4GammaLeptoNuclearPhys"), whose photonNuclear process
uses G4GammaNuclearXS by default: below 130 MeV the evaluated table of G4PARTICLEXS 4.2 ``gamma/inel18`` (0.5 MeV
steps, per element), from 130 to 150 MeV a straight line to G4PhotoNuclearCrossSection, above it
G4PhotoNuclearCrossSection. ``SIGMA_MB`` is that table converted from mm^2 to millibarn (Geant4 data, distributed under
the Geant4 Software License; see NOTICE). Above 130 MeV this module does not reproduce G4PhotoNuclearCrossSection: it
uses ``HIGH_MB``, a rough nuclear photoabsorption curve for A = 40 (quasi-deuteron region, the Delta resonance near
0.3 GeV, a shadowed ~0.11 mb per nucleon above 1 GeV). In the reference 95 % of photonuclear reactions come from
photons below 150 MeV (median 19.7 MeV, the giant dipole resonance), so this approximation moves little.
"""
import numpy as np

E_STEP_MEV = 0.5
SIGMA_MB = np.array([
    0, 0.00387977, 0.00987765, 0.0197339, 0.0344784, 0.0553228, 0.0837164, 0.121421, 0.170613, 0.234038, 0.315215,
    0.418733, 0.55065, 0.719037, 0.934777, 1.2127, 1.57319, 2.04417, 2.66256, 3.47171, 4.50528, 5.74316, 7.05665,
    8.26155, 9.34242, 10.5467, 12.2004, 14.5596, 17.7908, 21.9349, 26.7654, 31.6029, 35.371, 37.1787, 37.0054,
    35.6315, 33.9706, 32.616, 31.7984, 31.4978, 31.5681, 31.6385, 31.4559, 31.2733, 30.2708, 29.2683, 27.6076,
    25.9469, 24.1013, 22.2556, 20.5587, 18.8617, 17.435, 16.0083, 14.8534, 13.6986, 12.775, 11.8514, 11.1123,
    10.3732, 9.83404, 9.29484, 8.75564, 8.21644, 7.85485, 7.49326, 7.13166, 6.77007, 6.51961, 6.26915, 6.01868,
    5.76822, 5.59027, 5.41231, 5.23436, 5.05641, 4.92764, 4.79887, 4.67011, 4.54134, 4.447, 4.35266, 4.25832,
    4.16398, 4.0943, 4.02463, 3.95495, 3.88527, 3.83354, 3.78181, 3.73008, 3.67835, 3.63981, 3.60127, 3.56273,
    3.52418, 3.49538, 3.46657, 3.43776, 3.40895, 3.38733, 3.36571, 3.34409, 3.32247, 3.30613, 3.2898, 3.27347,
    3.25713, 3.24466, 3.23219, 3.21971, 3.20724, 3.19755, 3.18786, 3.17817, 3.16848, 3.16075, 3.15303, 3.1453,
    3.13758, 3.13121, 3.12483, 3.11846, 3.11208, 3.10659, 3.1011, 3.0956, 3.09011, 3.08515, 3.08018, 3.07522,
    3.07026, 3.06556, 3.06086, 3.05616, 3.05146, 3.04683, 3.0422, 3.03757, 3.03294, 3.02823, 3.02353, 3.01882,
    3.01411, 3.00922, 3.00433, 2.99944, 2.99455, 2.98941, 2.98427, 2.97912, 2.97398, 2.96853, 2.96309, 2.95764,
    2.95219, 2.94641, 2.94063, 2.93485, 2.92907, 2.92268, 2.91628, 2.90989, 2.9035, 2.8971, 2.89071, 2.88431,
    2.87792, 2.87153, 2.86513, 2.85789, 2.85064, 2.8434, 2.83616, 2.82891, 2.82167, 2.81442, 2.80718, 2.79994,
    2.79269, 2.7847, 2.77672, 2.76873, 2.76075, 2.75276, 2.74477, 2.73679, 2.7288, 2.72082, 2.71283, 2.70426,
    2.69569, 2.68713, 2.67856, 2.66999, 2.66142, 2.65286, 2.64429, 2.63572, 2.62715, 2.61809, 2.60902, 2.59995,
    2.59089, 2.58182, 2.57276, 2.56369, 2.55463, 2.54556, 2.53649, 2.52743, 2.51836, 2.5093, 2.50023, 2.49117,
    2.4821, 2.47303, 2.46397, 2.4549, 2.44584, 2.43676, 2.42768, 2.4186, 2.40952, 2.40044, 2.39136, 2.38228, 2.3732,
    2.36412, 2.35504, 2.34596, 2.33688, 2.3278, 2.31872, 2.30964, 2.30056, 2.29148, 2.2824, 2.27332, 2.26424,
    2.25591, 2.24758, 2.23924, 2.23091, 2.22258, 2.21424, 2.20591, 2.19758, 2.18924, 2.18091, 2.17258, 2.16424,
    2.15591, 2.14758, 2.13924, 2.13091, 2.12258, 2.11425, 2.10591, 2.09758])  # at E = 0.5, 1.0, ..., 130.0 MeV
E_TABLE_MEV = E_STEP_MEV * np.arange(1, SIGMA_MB.size + 1)

# (E [MeV], sigma [mb]) above the table: rough, log-log interpolated (see the module docstring)
HIGH_MB = np.array([(130.0, float(SIGMA_MB[-1])), (150.0, 2.2), (200.0, 4.0), (250.0, 9.0), (320.0, 15.0),
                    (450.0, 11.0), (600.0, 8.0), (800.0, 6.5), (1000.0, 5.5), (2000.0, 4.6), (1e5, 4.4)])


def xsec_mb(e_mev):
    """Photonuclear cross section of argon [mb] at photon energy e_mev [MeV]."""
    e = np.asarray(e_mev, float)
    low = np.interp(e, E_TABLE_MEV, SIGMA_MB, left=0.0)
    he = np.clip(e, HIGH_MB[0, 0], HIGH_MB[-1, 0])
    high = np.exp(np.interp(np.log(he), np.log(HIGH_MB[:, 0]), np.log(HIGH_MB[:, 1])))
    return np.where(e <= E_TABLE_MEV[-1], low, high)
