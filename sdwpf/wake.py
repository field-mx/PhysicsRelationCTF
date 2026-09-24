"""Auditable Jensen wake prior for the SDWPF turbine array.

Angles use the meteorological convention: degrees clockwise from true north
describe where wind comes from.  Coordinates are interpreted as local easting
and northing unless a single global rotation is explicitly supplied.
"""

from __future__ import annotations

import numpy as np


ROTOR_DIAMETER_M = 82.0
CT_WIND_SPEED = np.array(
    [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
     8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
     13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5,
     18.0, 18.5, 19.0, 19.5, 20.0],
    dtype=np.float64,
)
CT_VALUE = np.array(
    [1.10636, .997744, .914581, .845178, .789601, .791005, .792174,
     .793314, .79468, .79599, .796901, .770161, .736003, .705273,
     .670687, .590175, .480418, .40617, .349699, .304647, .268106,
     .237712, .212103, .190332, .17165, .155527, .141506, .129266,
     .11853, .109046, .100623, .09314, .086435, .080407, .074977],
    dtype=np.float64,
)


def actual_wind_from(ndir_deg: np.ndarray, wdir_deg: np.ndarray) -> np.ndarray:
    """Recover absolute wind-from bearing from nacelle and relative bearing."""
    return np.mod(np.asarray(ndir_deg) + np.asarray(wdir_deg), 360.0)


def circular_error_deg(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference, predicted minus reference."""
    return (np.asarray(predicted) - np.asarray(reference) + 180.0) % 360.0 - 180.0


def propagation_unit_vector(wind_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return downstream unit vector components in east/north coordinates."""
    wind_to = np.deg2rad(np.mod(np.asarray(wind_from_deg) + 180.0, 360.0))
    return np.sin(wind_to), np.cos(wind_to)


def jensen_ct(wind_speed: np.ndarray) -> np.ndarray:
    """Interpolate the published SL1500/82 thrust curve.

    The published 3 m/s entry exceeds one, while the classical Jensen formula
    requires ``sqrt(1-Ct)``.  It is therefore clipped just below one.  Wake is
    disabled outside the documented 3--25 m/s operating interval.
    """
    speed = np.asarray(wind_speed, dtype=np.float64)
    ct = np.interp(speed, CT_WIND_SPEED, CT_VALUE,
                   left=CT_VALUE[0], right=CT_VALUE[-1])
    valid = np.isfinite(speed) & (speed >= 3.0) & (speed <= 25.0)
    return np.where(valid, np.clip(ct, 0.0, 0.999), 0.0)


def jensen_wake_bias(
    xy: np.ndarray,
    wind_from_deg: np.ndarray,
    wind_speed: np.ndarray,
    kw: float,
    rotor_diameter: float = ROTOR_DIAMETER_M,
    coordinate_rotation_deg: float = 0.0,
) -> np.ndarray:
    """Build ``B[batch, destination, source]`` with hard Jensen wake support."""
    if kw <= 0:
        raise ValueError("kw must be positive")
    xy = np.asarray(xy, dtype=np.float64)
    direction = np.asarray(wind_from_deg, dtype=np.float64)
    speed = np.asarray(wind_speed, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape [turbine, 2]")
    if direction.ndim == 1:
        direction = direction[None]
    if speed.ndim == 1:
        speed = speed[None]
    if direction.shape != speed.shape or direction.shape[1] != len(xy):
        raise ValueError("wind direction/speed must have shape [batch, turbine]")

    # A positive coordinate rotation means the local +y axis is clockwise from
    # true north; subtract it to express the bearing in the local frame.
    east, north = propagation_unit_vector(direction - coordinate_rotation_deg)
    dr = xy[None, :, None, :] - xy[None, None, :, :]  # [1,dst,src,xy]
    d_parallel = dr[..., 0] * east[:, None, :] + dr[..., 1] * north[:, None, :]
    d_perp = np.abs(dr[..., 0] * north[:, None, :] - dr[..., 1] * east[:, None, :])
    wake_radius = rotor_diameter / 2.0 + kw * d_parallel
    support = (d_parallel > 0.0) & (d_perp <= wake_radius)
    support &= np.isfinite(direction)[:, None, :]

    ct = jensen_ct(speed)[:, None, :]
    denominator = (1.0 + 2.0 * kw * np.maximum(d_parallel, 0.0) / rotor_diameter) ** 2
    deficit = (1.0 - np.sqrt(1.0 - ct)) / denominator
    bias = np.where(support, deficit, 0.0)
    indices = np.arange(len(xy))
    bias[:, indices, indices] = 0.0
    return bias.astype(np.float32)
