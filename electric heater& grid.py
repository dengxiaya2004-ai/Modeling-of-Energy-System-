import numpy as np


# =========================
# 1. ELECTRIC HEATER CLASS
# =========================
class ElectricHeater:
    """
    Electric Heater model
    Converts electricity (MW_el) into heat (MW_th)
    """

    def __init__(self, P_max, efficiency=0.99):
        self.P_max = P_max  # Maximum electrical input [MW]
        self.efficiency = efficiency  # Conversion efficiency

    def dispatch(self, heat_demand, available_power):
        """
        Determine heater output based on demand and available electricity

        Parameters:
        heat_demand : float [MW_th]
        available_power : float [MW_el]

        Returns:
        P_eh : electricity used [MW_el]
        Q_eh : heat produced [MW_th]
        """

        # Maximum heat the heater can produce
        Q_max = self.P_max * self.efficiency

        # Step 1: meet demand within capacity
        Q_eh = min(heat_demand, Q_max)

        # Convert heat to required electricity
        P_eh = Q_eh / self.efficiency

        # Step 2: check if enough power is available
        if P_eh > available_power:
            P_eh = available_power
            Q_eh = P_eh * self.efficiency

        return P_eh, Q_eh


# =========================
# 2. GRID MODEL
# =========================
class Grid:
    """
    Grid model for electricity import
    """

    def __init__(self, price_import, P_max_import=np.inf):
        self.price_import = price_import  # €/MWh
        self.P_max_import = P_max_import  # MW

    def import_power(self, demand):
        """Limit grid import by capacity"""
        return min(demand, self.P_max_import)

    def compute_cost(self, P_grid, t):
        """Electricity cost at timestep t"""
        return P_grid * self.price_import[t]


# =========================
# 3. LOAD PROFILE (INDUSTRIAL)
# =========================
def generate_heat_demand(time_steps):
    """
    Industrial heat demand:
    10 MW during working hours (8:00–18:00)
    0 MW otherwise
    """

    hours = np.arange(time_steps) % 24
    heat_demand = np.zeros(time_steps)

    # Working hours: 8:00–18:00 (10 hours)
    heat_demand[(hours >= 8) & (hours < 18)] = 10

    return heat_demand


# =========================
# 4. ELECTRICITY PRICE (ITALY TOU)
# =========================
def generate_electricity_price(time_steps):
    """
    Italian-style Time-of-Use electricity price

    Daytime (08:00–20:00): high price
    Nighttime: low price
    """

    hours = np.arange(time_steps) % 24

    price = np.where(
        (hours >= 8) & (hours < 20),
        140,   # €/MWh (day)
        70     # €/MWh (night)
    )

    return price


# =========================
# 5. PV GENERATION (SIMPLE)
# =========================
def generate_pv_profile(time_steps, peak_power=4):
    """
    Simple synthetic PV profile (daytime only)
    """

    hours = np.arange(time_steps) % 24

    # PV only produces during daytime
    pv = peak_power * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))

    return pv


# =========================
# 6. MAIN SIMULATION
# =========================
def simulate_system(time_steps,
                    heat_demand,
                    pv_power_dc,
                    electricity_price,
                    heater,
                    grid,
                    inverter_efficiency=0.97):

    # Convert PV from DC to AC
    pv_power_ac = pv_power_dc * inverter_efficiency

    # Initialize result arrays
    P_EH = np.zeros(time_steps)
    Q_EH = np.zeros(time_steps)
    P_grid = np.zeros(time_steps)
    cost = np.zeros(time_steps)

    for t in range(time_steps):

        # Available PV power
        available_pv = pv_power_ac[t]

        # Step 1: use PV first
        P_eh_pv, Q_eh = heater.dispatch(
            heat_demand[t],
            available_pv
        )

        # Remaining heat demand
        remaining_heat = heat_demand[t] - Q_eh

        # Step 2: use grid if needed
        if remaining_heat > 0:

            # Convert heat demand to electricity
            P_needed = remaining_heat / heater.efficiency

            # Import from grid
            P_grid_t = grid.import_power(P_needed)

            # Convert to heat
            Q_extra = P_grid_t * heater.efficiency

            # Update totals
            Q_eh += Q_extra
            P_eh_total = P_eh_pv + P_grid_t

        else:
            P_grid_t = 0
            P_eh_total = P_eh_pv

        # Step 3: compute cost
        cost_t = grid.compute_cost(P_grid_t, t)

        # Store results
        P_EH[t] = P_eh_total
        Q_EH[t] = Q_eh
        P_grid[t] = P_grid_t
        cost[t] = cost_t

    return {
        "P_EH": P_EH,
        "Q_EH": Q_EH,
        "P_grid": P_grid,
        "cost": cost,
        "total_cost": np.sum(cost)
    }


# =========================
# 7. RUN SIMULATION
# =========================
if __name__ == "__main__":

    time_steps = 8760

    # Generate profiles
    heat_demand = generate_heat_demand(time_steps)
    electricity_price = generate_electricity_price(time_steps)
    pv_power_dc = generate_pv_profile(time_steps)

    # Create system components
    heater = ElectricHeater(P_max=10)
    grid = Grid(price_import=electricity_price)

    # Run simulation
    results = simulate_system(
        time_steps,
        heat_demand,
        pv_power_dc,
        electricity_price,
        heater,
        grid
    )

    # Print results
    print("Total yearly cost (€):", results["total_cost"])
    print("Average grid import (MW):", np.mean(results["P_grid"]))