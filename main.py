# main.py
# =========================================================
# Entry point — run this file to execute the full simulation.
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import params as P
import climate_data
import economics
import optimiser


def build_demand(climate: dict) -> np.ndarray:
    """Build the hourly thermal demand profile (8760 values)."""
    Q = np.zeros(climate['n_hours'])
    for i in range(climate['n_hours']):
        h = climate['hour_of_day'][i]
        if P.HOUR_START <= h < P.HOUR_END:
            Q[i] = P.LOAD_W
    return Q


def plot_results(opt_result: dict, Q_demand: np.ndarray, climate: dict):
    """Generate all output plots."""
    best     = opt_result['best']
    disp     = best['dispatch']
    months   = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

    # ---- Plot 1: LCOH surface (heatmap) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    lcoh = opt_result['lcoh_grid']
    n_stc_vals = opt_result['n_stc_values']
    e_tes_vals = [e/1e6 for e in opt_result['e_tes_values_Wh']]

    im = ax.contourf(e_tes_vals, n_stc_vals, lcoh,
                     levels=20, cmap='RdYlGn_r')
    plt.colorbar(im, ax=ax, label='LCOH [€/MWh_th]')
    # Mark optimal point
    opt_e   = best['E_tes_Wh'] / 1e6
    opt_n   = best['n_stc']
    ax.plot(opt_e, opt_n, 'w*', markersize=14, label=f'Optimum: {best["econ"]["LCOH"]:.1f} €/MWh')
    ax.set_xlabel('TES capacity [MWh]')
    ax.set_ylabel('Number of STC modules')
    ax.set_title('LCOH optimisation surface — STC field size vs TES capacity')
    ax.legend()
    plt.tight_layout()

    # ---- Plot 2: Monthly energy breakdown ----
    E_monthly = np.zeros((12, 4))
    for m in range(1, 13):
        idx = climate['month'] == m
        E_monthly[m-1, 0] = np.minimum(disp['Q_solar_W'][idx], Q_demand[idx]).sum() / 1e6
        E_monthly[m-1, 1] = disp['Q_tes_W'][idx].sum()      / 1e6
        E_monthly[m-1, 2] = disp['Q_hp_W'][idx].sum()       / 1e6
        E_monthly[m-1, 3] = disp['Q_elheater_W'][idx].sum() / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(12)
    w = 0.6
    ax.bar(x, E_monthly[:,0], w, label='STC (direct)',   color='#F5A623')
    ax.bar(x, E_monthly[:,1], w, bottom=E_monthly[:,0],  label='TES',   color='#4A90D9')
    ax.bar(x, E_monthly[:,2], w, bottom=E_monthly[:,0:2].sum(1), label='Heat pump', color='#7ED321')
    ax.bar(x, E_monthly[:,3], w, bottom=E_monthly[:,0:3].sum(1), label='El. heater',color='#D0021B')
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylabel('Thermal energy [MWh]')
    ax.set_title('Monthly thermal energy breakdown — optimal configuration')
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    # ---- Plot 3: CAPEX breakdown pie chart ----
    econ = best['econ']
    labels  = ['STC', 'TES', 'HP (placeholder)', 'El. heater (placeholder)']
    values  = [econ['capex_stc'], econ['capex_tes'], 0, 0]
    values  = [v for v in values if v > 0]
    labels  = [l for l, v in zip(labels, [econ['capex_stc'], econ['capex_tes'], 0, 0]) if v > 0]
    colors  = ['#F5A623', '#4A90D9', '#7ED321', '#D0021B'][:len(values)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'CAPEX breakdown — Total: {econ["capex_total"]/1e6:.2f} M€')
    plt.tight_layout()

    # ---- Plot 4: Typical week SOC and power ----
    idx_jun = np.where(climate['month'] == 6)[0][:168]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.fill_between(range(168), Q_demand[idx_jun]/1e6, alpha=0.15, color='gray', label='Demand')
    ax1.fill_between(range(168), np.minimum(disp['Q_solar_W'][idx_jun], Q_demand[idx_jun])/1e6,
                     alpha=0.8, color='#F5A623', label='STC')
    ax1.fill_between(range(168), disp['Q_tes_W'][idx_jun]/1e6,
                     alpha=0.8, color='#4A90D9', label='TES')
    ax1.plot(range(168), Q_demand[idx_jun]/1e6, 'k--', lw=1.2, label='Demand')
    ax1.set_ylabel('Thermal power [MW]'); ax1.legend(loc='upper right'); ax1.grid(alpha=0.4)
    ax1.set_title('Typical summer week (June) — hourly power balance')

    ax2.plot(range(168), disp['SOC_TES'][idx_jun]*100, 'b-', lw=1.5)
    ax2.axhline(P.SOC_MIN*100, color='r', ls='--', label='Min SOC')
    ax2.axhline(100, color='g', ls='--', label='Max SOC')
    ax2.set_ylabel('TES SOC [%]'); ax2.set_xlabel('Hour of week')
    ax2.set_xticks(range(0, 169, 24))
    ax2.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun',''])
    ax2.legend(); ax2.grid(alpha=0.4); ax2.set_ylim(0, 110)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    print("=== Hybrid solar thermal system — Seville, Spain ===\n")

    # 1. Load climate data
    climate  = climate_data.load_climate_data('seville_tmy.csv')

    # 2. Build demand profile
    Q_demand = build_demand(climate)
    E_annual = Q_demand.sum() / 1e6
    print(f"\n[Demand] Annual thermal demand: {E_annual:.1f} MWh"
          f"  ({Q_demand.sum()/1e9:.2f} GWh)\n")

    # 3. Run optimiser (sweeps STC and TES sizes, minimises LCOH)
    opt_result = optimiser.run(climate, Q_demand)

    # 4. Print full economic summary for optimal configuration
    economics.print_summary(
        opt_result['best']['econ'],
        opt_result['best']['dispatch']
    )

    # 5. Generate plots
    plot_results(opt_result, Q_demand, climate)