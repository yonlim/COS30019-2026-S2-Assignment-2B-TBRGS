"""Task 3 (part 2): turning a predicted traffic flow into a travel time.

Owner: Jonathan Yu Yung Jing (105815665)

Follows "Traffic Flow to Travel Time Conversion v1.0" (unit resources):

    flow = A * speed^2 + B * speed           (a simplified fundamental diagram)
    A = -capacity_flow / capacity_speed^2  = -1500 / 32^2 = -1.46484375
    B = -2 * capacity_speed * A           =  93.75

The note's text prints A as -1.4648375 (one digit dropped); its graph uses
-1.46484375, which is what the 1500 veh/h and 32 km/h turning point gives.

Solving for speed gives two answers. The road is assumed to be under capacity
(the note's assumption), so the larger, "green" answer is used:

    speed = (-B - sqrt(B^2 + 4 * A * flow)) / (2 * A)

capped at the 60 km/h speed limit (flows up to about 351 veh/h). A predicted
flow at or above capacity has no real answer, so its speed is set to the
capacity speed, 32 km/h. Each link then costs

    time = distance / speed + 30 s delay for the controlled intersection at its end.

The flow is the predicted 15-minute count times 4 (vehicles per hour).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TravelTimeModel:
    speed_limit_kmh: float = 60.0
    capacity_flow_vph: float = 1500.0
    capacity_speed_kmh: float = 32.0
    intersection_delay_s: float = 30.0

    def __post_init__(self):
        if not 0 < self.capacity_speed_kmh <= self.speed_limit_kmh:
            raise ValueError("capacity speed must be positive and not above the speed limit")
        if self.capacity_flow_vph <= 0 or self.intersection_delay_s < 0:
            raise ValueError("capacity flow must be positive and the delay non-negative")

    @property
    def a(self) -> float:
        return -self.capacity_flow_vph / self.capacity_speed_kmh ** 2

    @property
    def b(self) -> float:
        return -2 * self.capacity_speed_kmh * self.a

    def flow_from_speed(self, speed_kmh: float) -> float:
        """The note's curve: flow (veh/h) at a given speed (km/h)."""
        return self.a * speed_kmh ** 2 + self.b * speed_kmh

    def speed_from_flow(self, flow_vph: float) -> float:
        """Speed (km/h) on an under-capacity road carrying flow_vph vehicles per hour."""
        if flow_vph is None or not math.isfinite(flow_vph) or flow_vph <= 0:
            return self.speed_limit_kmh
        if flow_vph >= self.capacity_flow_vph:
            return self.capacity_speed_kmh
        discriminant = self.b ** 2 + 4 * self.a * flow_vph
        speed = (-self.b - math.sqrt(max(discriminant, 0.0))) / (2 * self.a)
        return min(speed, self.speed_limit_kmh)

    def link_time_s(self, distance_km: float, flow_vph: float) -> float:
        """Seconds to drive a link of distance_km carrying flow_vph, plus the intersection delay."""
        if distance_km < 0:
            raise ValueError("distance cannot be negative")
        return distance_km / self.speed_from_flow(flow_vph) * 3600 + self.intersection_delay_s

    def free_flow_seconds_per_km(self) -> float:
        """Seconds per km at the speed limit: the fastest any link can be driven."""
        return 3600 / self.speed_limit_kmh


def hourly_flow(count_per_15_min: float) -> float:
    """Vehicles per hour from a 15-minute count."""
    return 4.0 * float(count_per_15_min)
