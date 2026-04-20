import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def simulate_heat_pump_engine(
    Q_ST_hourly: np.ndarray,
    Q_demand_hourly: np.full(8760, 10e6),
    T_cold_HP: float = 70.0,
    T_hot_HP: float = 157.0,
    T_cold_HE: float = 40.0,
    T_hot_HE: float = 157.0,
    COP_real: float = 2.32,
    epsilon_HE: float = 0.20,
    Q_waste_max: float = 5e6,
    # ── Economic parameters ───────────────────────────────────────
    electricity_price: float = 70.0,       # €/MWh, Enerin Annex 58
    gas_price: float = 70.0,               # €/MWh, Enerin Annex 58
    capex_per_kW: float = 700.0,           # €/kW, mid of 600-800, Enerin Annex 58
    maintenance_per_MWh: float = 10.0,     # €/MWh recycled heat, Enerin Annex 58
    boiler_efficiency: float = 0.85,       # natural gas boiler baseline
    system_lifetime: float = 20.0,         # years, Enerin Annex 58
    discount_rate: float = 0.08,           # 8% typical industrial project
    P_HP_kW: float = 1000.0,              # installed HP capacity [kW]
    electricity_sell_price: float = 50.0,  # €/MWh, revenue from engine electricity
    plot: bool = True,
    save_fig: bool = True,
    fig_path: str = 'hp_engine_results.png'
) -> dict:
    """
    Simulate hourly operation of HTHP and Stirling Heat Engine,
    with full economic analysis (LCOH, NPV, IRR, payback period).

    Heat Pump parameters: Enerin HoegTemp, IEA HPT Annex 58, Table 1
    Economic parameters:  Enerin HoegTemp, IEA HPT Annex 58, Project Example

    Parameters
    ----------
    Q_ST_hourly        : np.ndarray  Hourly ST heat output [Wh], shape (8760,)
    Q_demand_hourly    : np.ndarray  Hourly industrial heat demand [Wh], shape (8760,)
    T_cold_HP          : float       HP cold side temperature [C], default 70
    T_hot_HP           : float       HP hot side temperature [C], default 157
    T_cold_HE          : float       Engine cold side temperature [C], default 40
    T_hot_HE           : float       Engine hot side temperature [C], default 157
    COP_real           : float       Actual HP COP [-], default 2.32
    epsilon_HE         : float       Engine Carnot correction [-], default 0.20
    Q_waste_max        : float       Max waste heat available [W], default 5e6
    electricity_price  : float       Electricity cost [EUR/MWh], default 70
    gas_price          : float       Natural gas price [EUR/MWh], default 70
    capex_per_kW       : float       Capital cost [EUR/kW], default 700
    maintenance_per_MWh: float       Maintenance cost [EUR/MWh heat], default 10
    boiler_efficiency  : float       Gas boiler baseline efficiency, default 0.85
    system_lifetime    : float       System lifetime [years], default 20
    discount_rate      : float       Discount rate [-], default 0.08
    P_HP_kW            : float       Installed HP capacity [kW], default 1000
    electricity_sell_price: float    Engine electricity sell price [EUR/MWh], default 50
    plot               : bool        Generate plots, default True
    save_fig           : bool        Save figure, default True
    fig_path           : str         Figure save path

    Returns
    -------
    dict with keys:
        hourly arrays, annual, monthly, performance, economics
    """

    # ── 1. Temperature conversion ─────────────────────────────────
    T_cold_HP_K = T_cold_HP + 273.15
    T_hot_HP_K  = T_hot_HP  + 273.15
    T_cold_HE_K = T_cold_HE + 273.15
    T_hot_HE_K  = T_hot_HE  + 273.15

    # ── 2. Performance parameters ─────────────────────────────────
    COP_carnot = T_hot_HP_K / (T_hot_HP_K - T_cold_HP_K)
    epsilon_HP = COP_real / COP_carnot
    eta_carnot = 1.0 - T_cold_HE_K / T_hot_HE_K
    eta_real   = epsilon_HE * eta_carnot

    # ── 3. Hourly simulation arrays ───────────────────────────────
    Q_HP_hourly     = np.zeros(8760)
    W_elec_hourly   = np.zeros(8760)
    Q_waste_hourly  = np.zeros(8760)
    W_engine_hourly = np.zeros(8760)
    Q_excess_hourly = np.zeros(8760)
    Q_reject_hourly = np.zeros(8760)

    # ── 4. Main hourly loop ───────────────────────────────────────
    for t in range(8760):
        Q_ST  = Q_ST_hourly[t]
        Q_dem = Q_demand_hourly[t]

        if Q_dem == 0:
            continue

        if Q_ST < Q_dem:
            # Heat Pump mode
            Q_HP              = Q_dem - Q_ST
            W_elec            = Q_HP / COP_real
            Q_waste_used      = min(Q_HP - W_elec, Q_waste_max)
            Q_HP_hourly[t]    = Q_HP
            W_elec_hourly[t]  = W_elec
            Q_waste_hourly[t] = Q_waste_used
        else:
            # Heat Engine mode
            Q_excess              = Q_ST - Q_dem
            W_out                 = eta_real * Q_excess
            Q_reject              = Q_excess - W_out
            Q_excess_hourly[t]    = Q_excess
            W_engine_hourly[t]    = W_out
            Q_reject_hourly[t]    = Q_reject

    # ── 5. Annual summary ─────────────────────────────────────────
    annual = {
        'HP_heat_MWh'    : float(np.sum(Q_HP_hourly)     / 1e6),
        'HP_elec_MWh'    : float(np.sum(W_elec_hourly)   / 1e6),
        'waste_heat_MWh' : float(np.sum(Q_waste_hourly)  / 1e6),
        'engine_elec_MWh': float(np.sum(W_engine_hourly) / 1e6),
        'excess_heat_MWh': float(np.sum(Q_excess_hourly) / 1e6),
        'heat_reject_MWh': float(np.sum(Q_reject_hourly) / 1e6),
        'HP_hours'       : int(np.sum(Q_HP_hourly > 0)),
        'engine_hours'   : int(np.sum(W_engine_hourly > 0)),
    }
    annual['COP_effective'] = (
        annual['HP_heat_MWh'] / annual['HP_elec_MWh']
        if annual['HP_elec_MWh'] > 0 else 0.0
    )

    # ── 6. Monthly breakdown ──────────────────────────────────────
    days_per_month  = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [d * 24 for d in days_per_month]
    HP_monthly      = np.zeros(12)
    engine_monthly  = np.zeros(12)
    h = 0
    for m in range(12):
        idx               = slice(h, h + hours_per_month[m])
        HP_monthly[m]     = np.sum(Q_HP_hourly[idx])     / 1e6
        engine_monthly[m] = np.sum(W_engine_hourly[idx]) / 1e6
        h += hours_per_month[m]

    monthly = {
        'HP_heat_MWh'    : HP_monthly,
        'engine_elec_MWh': engine_monthly,
    }

    # ── 7. Performance parameters ─────────────────────────────────
    performance = {
        'COP_carnot': round(COP_carnot, 3),
        'COP_real'  : round(COP_real,   3),
        'epsilon_HP': round(epsilon_HP, 3),
        'eta_carnot': round(eta_carnot, 3),
        'eta_real'  : round(eta_real,   3),
        'epsilon_HE': round(epsilon_HE, 3),
    }

    # ── 8. Economic analysis ──────────────────────────────────────

    # CAPEX
    capex = capex_per_kW * P_HP_kW                        # €

    # Annual costs
    cost_electricity = annual['HP_elec_MWh'] * electricity_price       # € HP electricity
    cost_maintenance = annual['HP_heat_MWh'] * maintenance_per_MWh     # € maintenance
    cost_opex        = cost_electricity + cost_maintenance              # € total OPEX

    # Annual revenue / savings
    # Saving 1: avoided gas boiler cost for HP heat output
    gas_equiv_cost   = (annual['HP_heat_MWh'] / boiler_efficiency) * gas_price
    saving_vs_boiler = gas_equiv_cost - cost_electricity - cost_maintenance

    # Saving 2: revenue from engine electricity sold
    revenue_engine   = annual['engine_elec_MWh'] * electricity_sell_price

    # Net annual benefit
    net_annual_benefit = saving_vs_boiler + revenue_engine              # €

    # Simple payback period
    payback_simple = capex / net_annual_benefit if net_annual_benefit > 0 else float('inf')

    # LCOH (Levelised Cost of Heat)
    # LCOH = (CAPEX * CRF + annual OPEX) / annual heat output
    # Capital Recovery Factor
    if discount_rate > 0:
        CRF  = (discount_rate * (1 + discount_rate) ** system_lifetime) / \
               ((1 + discount_rate) ** system_lifetime - 1)
    else:
        CRF  = 1.0 / system_lifetime

    annual_capital_cost = capex * CRF                                   # €/year
    total_annual_cost   = annual_capital_cost + cost_opex               # €/year
    LCOH = (total_annual_cost / annual['HP_heat_MWh']
            if annual['HP_heat_MWh'] > 0 else float('inf'))             # €/MWh

    # Baseline LCOH from gas boiler
    LCOH_boiler = gas_price / boiler_efficiency                         # €/MWh

    # NPV over lifetime
    npv = -capex
    for year in range(1, int(system_lifetime) + 1):
        npv += net_annual_benefit / (1 + discount_rate) ** year         # €

    # IRR (Internal Rate of Return) via bisection method
    def npv_at_rate(r):
        return -capex + sum(
            net_annual_benefit / (1 + r) ** y
            for y in range(1, int(system_lifetime) + 1)
        )

    irr = None
    if net_annual_benefit > 0:
        try:
            lo, hi = 0.0, 5.0
            for _ in range(100):
                mid = (lo + hi) / 2
                if npv_at_rate(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            irr = round((lo + hi) / 2 * 100, 2)    # as percentage
        except Exception:
            irr = None

    economics = {
        'capex_EUR'            : round(capex,              2),
        'cost_electricity_EUR' : round(cost_electricity,   2),
        'cost_maintenance_EUR' : round(cost_maintenance,   2),
        'cost_opex_EUR'        : round(cost_opex,          2),
        'saving_vs_boiler_EUR' : round(saving_vs_boiler,   2),
        'revenue_engine_EUR'   : round(revenue_engine,     2),
        'net_annual_benefit_EUR': round(net_annual_benefit,2),
        'payback_simple_years' : round(payback_simple,     2),
        'LCOH_EUR_per_MWh'     : round(LCOH,               2),
        'LCOH_boiler_EUR_per_MWh': round(LCOH_boiler,      2),
        'NPV_EUR'              : round(npv,                 2),
        'IRR_pct'              : irr,
        'CRF'                  : round(CRF,                 4),
    }

    # ── 9. Plotting ───────────────────────────────────────────────
    if plot:
        months_label = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

        fig = plt.figure(figsize=(18, 22))
        fig.suptitle(
            'Heat Pump & Stirling Engine — Annual Simulation & Economic Results\n'
            'Seville | 10 MW Industrial Load | Enerin HoegTemp',
            fontsize=14, fontweight='bold', y=0.99
        )
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

        # Plot 1: Monthly bar
        ax1 = fig.add_subplot(gs[0, 0])
        x, w = np.arange(12), 0.35
        ax1.bar(x - w/2, monthly['HP_heat_MWh'],     w,
                label='HP Heat Output',     color='#E07B39')
        ax1.bar(x + w/2, monthly['engine_elec_MWh'], w,
                label='Engine Elec Output', color='#3A7EBF')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months_label)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Energy (MWh)')
        ax1.set_title('Monthly HP Heat & Engine Electricity Output')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.4)

        # Plot 2: Pie chart
        ax2    = fig.add_subplot(gs[0, 1])
        vals   = [annual['HP_heat_MWh'],    annual['engine_elec_MWh'],
                  annual['waste_heat_MWh'], annual['heat_reject_MWh']]
        lbls   = ['HP Heat\nOutput', 'Engine Elec\nOutput',
                  'Waste Heat\nUsed', 'Engine Heat\nRejected']
        colors = ['#E07B39', '#3A7EBF', '#5BAD6F', '#C0392B']
        non_zero = [(v, l, c) for v, l, c in zip(vals, lbls, colors) if v > 0]
        if non_zero:
            v_p, l_p, c_p = zip(*non_zero)
            ax2.pie(v_p, labels=l_p, colors=c_p,
                    autopct='%1.1f%%', startangle=140,
                    textprops={'fontsize': 9})
        ax2.set_title('Annual Energy Distribution')

        # Plot 3: First week stack
        ax3     = fig.add_subplot(gs[1, :])
        t_week  = np.arange(168)
        Q_ST_kW = Q_ST_hourly[:168]     / 1e3
        Q_HP_kW = Q_HP_hourly[:168]     / 1e3
        Q_dm_kW = Q_demand_hourly[:168] / 1e3
        Q_en_kW = W_engine_hourly[:168] / 1e3
        ax3.fill_between(t_week, 0, Q_ST_kW,
                         alpha=0.6, color='#F4D03F', label='ST Output (kWh)')
        ax3.fill_between(t_week, Q_ST_kW, Q_ST_kW + Q_HP_kW,
                         alpha=0.6, color='#E07B39', label='HP Output (kWh)')
        ax3.plot(t_week, Q_dm_kW,
                 'r--', linewidth=1.8, label='Demand (kWh)')
        ax3.fill_between(t_week, 0, -Q_en_kW,
                         alpha=0.5, color='#3A7EBF', label='Engine Elec Out (kWh)')
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_xlabel('Hour (first 7 days)')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('Heat Supply Stack vs Demand — First Week')
        ax3.legend(loc='upper right')
        ax3.grid(alpha=0.3)

        # Plot 4: COP sensitivity
        ax4        = fig.add_subplot(gs[2, 0])
        T_cold_rng = np.arange(30, 120, 5)
        COP_curve  = np.clip(
            performance['epsilon_HP'] * T_hot_HP_K /
            (T_hot_HP_K - (T_cold_rng + 273.15)), 0, 10
        )
        ax4.plot(T_cold_rng, COP_curve, 'b-o', markersize=4, linewidth=1.5)
        ax4.axvline(x=70, color='red',    linestyle='--', linewidth=1.2,
                    label='Project T_cold = 70°C')
        ax4.axhline(y=COP_real, color='orange', linestyle='--', linewidth=1.2,
                    label=f'COP = {COP_real:.2f}')
        ax4.set_xlabel('Waste Heat Temperature T_cold (°C)')
        ax4.set_ylabel('Real COP [-]')
        ax4.set_title('HP COP vs Waste Heat Temperature')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.4)

        # Plot 5: LCOH comparison
        ax5 = fig.add_subplot(gs[2, 1])
        lcoh_labels = ['HP System\n(this project)', 'Gas Boiler\n(baseline)']
        lcoh_vals   = [economics['LCOH_EUR_per_MWh'],
                       economics['LCOH_boiler_EUR_per_MWh']]
        lcoh_colors = ['#3A7EBF', '#C0392B']
        bars5 = ax5.bar(lcoh_labels, lcoh_vals, color=lcoh_colors,
                        alpha=0.85, edgecolor='white', width=0.4)
        for bar, val in zip(bars5, lcoh_vals):
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(lcoh_vals) * 0.02,
                     f'{val:.1f} €/MWh',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax5.set_ylabel('LCOH (€/MWh)')
        ax5.set_title('Levelised Cost of Heat — HP vs Gas Boiler')
        ax5.grid(axis='y', alpha=0.4)

        # Plot 6: Cumulative NPV over lifetime
        ax6  = fig.add_subplot(gs[3, :])
        years      = np.arange(0, int(system_lifetime) + 1)
        cum_npv    = np.zeros(len(years))
        cum_npv[0] = -capex
        for y in range(1, len(years)):
            cum_npv[y] = cum_npv[y-1] + \
                         net_annual_benefit / (1 + discount_rate) ** y
        ax6.plot(years, cum_npv / 1e3, 'g-o', markersize=4, linewidth=1.8)
        ax6.axhline(0, color='black', linewidth=1.0, linestyle='--')
        ax6.fill_between(years, cum_npv / 1e3, 0,
                         where=(cum_npv >= 0),
                         alpha=0.25, color='green', label='Positive NPV')
        ax6.fill_between(years, cum_npv / 1e3, 0,
                         where=(cum_npv < 0),
                         alpha=0.25, color='red',   label='Negative NPV')
        if payback_simple <= system_lifetime:
            ax6.axvline(x=payback_simple, color='blue',
                        linestyle='--', linewidth=1.2,
                        label=f'Payback = {payback_simple:.1f} yr')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Cumulative NPV (k€)')
        ax6.set_title(f'Cumulative NPV over {int(system_lifetime)}-Year Lifetime  '
                      f'(Discount rate = {discount_rate*100:.0f}%)')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # KPI + Economics text box
        irr_str = f"{economics['IRR_pct']:.1f}%" if economics['IRR_pct'] else 'N/A'
        kpi_text = (
            f"  Performance                      Economics\n"
            f"  {'─'*28}    {'─'*28}\n"
            f"  COP Carnot:    {performance['COP_carnot']:.2f}          "
            f"CAPEX:           {economics['capex_EUR']/1e3:.0f} k€\n"
            f"  COP Real:      {performance['COP_real']:.2f}          "
            f"Annual OPEX:     {economics['cost_opex_EUR']/1e3:.0f} k€/yr\n"
            f"  COP Effective: {annual['COP_effective']:.2f}          "
            f"Net Benefit:     {economics['net_annual_benefit_EUR']/1e3:.0f} k€/yr\n"
            f"  eta Real (HE): {performance['eta_real']*100:.1f}%           "
            f"LCOH (HP):       {economics['LCOH_EUR_per_MWh']:.1f} €/MWh\n"
            f"  HP Hours/yr:   {annual['HP_hours']} h         "
            f"LCOH (Boiler):   {economics['LCOH_boiler_EUR_per_MWh']:.1f} €/MWh\n"
            f"  Engine Hrs/yr: {annual['engine_hours']} h         "
            f"NPV:             {economics['NPV_EUR']/1e3:.0f} k€\n"
            f"                                   "
            f"IRR:             {irr_str}\n"
            f"                                   "
            f"Payback:         {economics['payback_simple_years']:.1f} yr"
        )
        fig.text(0.5, 0.002, kpi_text, ha='center', va='bottom',
                 fontsize=8.5, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#EBF3FB',
                           edgecolor='#2E74B5', alpha=0.9))

        if save_fig:
            plt.savefig(fig_path, dpi=150,
                        bbox_inches='tight', facecolor='white')
            print(f"Figure saved as {fig_path}")

        plt.show()

    # ── 10. Return ────────────────────────────────────────────────
    return {
        'Q_HP_hourly'    : Q_HP_hourly,
        'W_elec_hourly'  : W_elec_hourly,
        'Q_waste_hourly' : Q_waste_hourly,
        'W_engine_hourly': W_engine_hourly,
        'Q_excess_hourly': Q_excess_hourly,
        'Q_reject_hourly': Q_reject_hourly,
        'annual'         : annual,
        'monthly'        : monthly,
        'performance'    : performance,
        'economics'      : economics,
    }
