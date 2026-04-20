import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def simulate_heat_pump_engine(
    Q_ST_hourly: np.ndarray,
    Q_demand_hourly:  np.full(8760, 10e6),
    T_cold_HP: float = 70.0,
    T_hot_HP: float = 157.0,
    T_cold_HE: float = 40.0,
    T_hot_HE: float = 157.0,
    COP_real: float = 2.32,
    epsilon_HE: float = 0.20,
    Q_waste_max: float = 5e6,
    plot: bool = True,
    save_fig: bool = True,
    fig_path: str = 'hp_engine_results.png'
) -> dict:
    """
    Simulate hourly operation of High-Temperature Heat Pump (HTHP)
    and Stirling Heat Engine in a hybrid solar industrial heat system.
    Optionally generates and saves result plots.

    Heat Pump parameters based on:
    Enerin HoegTemp, IEA HPT Annex 58, Table 1
    (Tsource=80->50C, Tsink=154->160C, COP=2.2-2.45)

    Parameters
    ----------
    Q_ST_hourly     : np.ndarray  Hourly ST heat output [Wh], shape (8760,)
    Q_demand_hourly : np.ndarray  Hourly industrial heat demand [Wh], shape (8760,)
    T_cold_HP       : float       Heat pump cold side temperature [C], default 70
    T_hot_HP        : float       Heat pump hot side temperature [C], default 157
    T_cold_HE       : float       Heat engine cold side temperature [C], default 40
    T_hot_HE        : float       Heat engine hot side temperature [C], default 157
    COP_real        : float       Actual heat pump COP [-], default 2.32
    epsilon_HE      : float       Heat engine Carnot correction factor [-], default 0.20
    Q_waste_max     : float       Max available waste heat [W], default 5e6
    plot            : bool        Whether to generate plots, default True
    save_fig        : bool        Whether to save figure to file, default True
    fig_path        : str         File path for saved figure, default 'hp_engine_results.png'

    Returns
    -------
    dict with keys:
        'Q_HP_hourly'     : np.ndarray  Heat pump heat output [Wh]
        'W_elec_hourly'   : np.ndarray  Heat pump electricity input [Wh]
        'Q_waste_hourly'  : np.ndarray  Waste heat consumed [Wh]
        'W_engine_hourly' : np.ndarray  Engine electricity output [Wh]
        'Q_excess_hourly' : np.ndarray  Excess ST heat to engine [Wh]
        'Q_reject_hourly' : np.ndarray  Engine heat rejection [Wh]
        'annual'          : dict        Annual summary statistics
        'monthly'         : dict        Monthly breakdown
        'performance'     : dict        COP and efficiency values
    """

    # ── 1. Temperature conversion ─────────────────────────────────
    T_cold_HP_K = T_cold_HP + 273.15
    T_hot_HP_K  = T_hot_HP  + 273.15
    T_cold_HE_K = T_cold_HE + 273.15
    T_hot_HE_K  = T_hot_HE  + 273.15

    # ── 2. Derived performance parameters ─────────────────────────
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
            # Heat Pump mode: ST insufficient, HP covers the gap
            Q_HP              = Q_dem - Q_ST
            W_elec            = Q_HP / COP_real
            Q_waste_used      = min(Q_HP - W_elec, Q_waste_max)
            Q_HP_hourly[t]    = Q_HP
            W_elec_hourly[t]  = W_elec
            Q_waste_hourly[t] = Q_waste_used
        else:
            # Heat Engine mode: ST excess drives engine
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
        idx             = slice(h, h + hours_per_month[m])
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

    # ── 8. Plotting ───────────────────────────────────────────────
    if plot:
        months_label = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

        fig = plt.figure(figsize=(16, 18))
        fig.suptitle(
            'Heat Pump & Stirling Engine — Annual Simulation Results\n'
            'Seville | 10 MW Industrial Load | Enerin HoegTemp',
            fontsize=14, fontweight='bold', y=0.98
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        # Plot 1: Monthly bar chart
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
        Q_ST_kW = Q_ST_hourly[:168]        / 1e3
        Q_HP_kW = Q_HP_hourly[:168]        / 1e3
        Q_dm_kW = Q_demand_hourly[:168]    / 1e3
        Q_en_kW = W_engine_hourly[:168]    / 1e3
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
            (T_hot_HP_K - (T_cold_rng + 273.15)),
            0, 10
        )
        ax4.plot(T_cold_rng, COP_curve, 'b-o', markersize=4, linewidth=1.5)
        ax4.axvline(x=70,  color='red',    linestyle='--', linewidth=1.2,
                    label='Project T_cold = 70°C')
        ax4.axhline(y=COP_real, color='orange', linestyle='--', linewidth=1.2,
                    label=f'COP = {COP_real:.2f}')
        ax4.set_xlabel('Waste Heat Temperature T_cold (°C)')
        ax4.set_ylabel('Real COP [-]')
        ax4.set_title('HP COP vs Waste Heat Temperature')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.4)

        # Plot 5: Annual summary bar
        ax5      = fig.add_subplot(gs[2, 1])
        cat      = ['HP Heat\nOutput', 'HP Elec\nInput', 'Waste Heat\nUsed',
                    'Engine Elec\nOutput', 'Excess ST\nHeat', 'Heat\nRejected']
        vals_bar = [annual['HP_heat_MWh'],    annual['HP_elec_MWh'],
                    annual['waste_heat_MWh'], annual['engine_elec_MWh'],
                    annual['excess_heat_MWh'],annual['heat_reject_MWh']]
        bar_colors = ['#E07B39','#E07B39','#5BAD6F',
                      '#3A7EBF','#F4D03F','#C0392B']
        bars = ax5.bar(cat, vals_bar, color=bar_colors,
                       alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals_bar):
            if val > 0:
                ax5.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(vals_bar) * 0.01,
                         f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        ax5.set_ylabel('Energy (MWh/year)')
        ax5.set_title('Annual Energy Summary')
        ax5.tick_params(axis='x', labelsize=8)
        ax5.grid(axis='y', alpha=0.4)

        # KPI text box
        kpi_text = (
            f"Key Performance Indicators\n"
            f"{'─'*32}\n"
            f"COP Carnot:      {performance['COP_carnot']:.2f}\n"
            f"COP Real:        {performance['COP_real']:.2f}  "
            f"(epsilon = {performance['epsilon_HP']:.2f})\n"
            f"COP Effective:   {annual['COP_effective']:.2f}\n"
            f"eta Carnot (HE): {performance['eta_carnot']*100:.1f}%\n"
            f"eta Real (HE):   {performance['eta_real']*100:.1f}%\n"
            f"HP Hours/year:   {annual['HP_hours']} h\n"
            f"Engine Hours/yr: {annual['engine_hours']} h"
        )
        fig.text(0.5, 0.005, kpi_text, ha='center', va='bottom',
                 fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#EBF3FB',
                           edgecolor='#2E74B5', alpha=0.9))

        if save_fig:
            plt.savefig(fig_path, dpi=150,
                        bbox_inches='tight', facecolor='white')
            print(f"Figure saved as {fig_path}")

        plt.show()

    # ── 9. Return ─────────────────────────────────────────────────
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
    }
