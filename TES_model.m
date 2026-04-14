function [Q_TES_out, Q_residual, SOC] = TES_model(Q_solar, Q_demand, Q_residual_in, params)
%% TES_model.m
% Simulates the hot water thermal energy storage tank (TES) over a full year.
%
% INPUT:
%   Q_solar       : [W] 8760x1 — Thermal power from the STC field
%   Q_demand      : [W] 8760x1 — Original process demand (needed for surplus calc)
%   Q_residual_in : [W] 8760x1 — Remaining demand after STC (from STC_model.m)
%   params        : struct from params_system.m (P)
%
% OUTPUT:
%   Q_TES_out  : [W] 8760x1 — Thermal power delivered by TES to the process
%   Q_residual : [W] 8760x1 — Remaining demand after TES contribution
%   SOC        : [-] 8760x1 — State of charge at each hour (0=empty, 1=full)

    n = length(Q_residual_in);

    Q_TES_out  = zeros(n, 1);
    Q_residual = zeros(n, 1);
    SOC        = zeros(n, 1);

    E_max   = params.E_TES_J;
    SOC_now = params.SOC_initial;

    for i = 1:n
        E_stored = SOC_now * E_max;

        % Passive thermal losses this hour [J]
        E_loss = params.TES_loss_frac * E_stored * 3600;

        % TRUE surplus = solar production vs original demand
        % Positive: STC produces more than needed  -> charge TES
        % Negative: STC produces less than needed  -> discharge TES
        Q_surplus = Q_solar(i) - Q_demand(i);   % FIX: use Q_demand, not Q_residual_in

        if Q_surplus >= 0
            %% Case 1: STC covers full demand AND has surplus -> charge TES
            E_charge = Q_surplus * 3600;
            E_new    = E_stored + E_charge - E_loss;
            E_new    = min(E_new, E_max * params.SOC_max);
            E_new    = max(E_new, 0);

            Q_TES_out(i)  = 0;
            Q_residual(i) = 0;  % STC already covered everything

        else
            %% Case 2: STC does not cover full demand -> discharge TES
            E_deficit   = Q_residual_in(i) * 3600;   % [J] unmet demand after STC
            E_available = max(E_stored - params.SOC_min * E_max, 0);

            E_discharge = min(E_deficit, E_available);

            E_new = E_stored - E_discharge - E_loss;
            E_new = max(E_new, 0);

            Q_TES_out(i) = E_discharge / 3600;  % [W]

            % FIX: residual is simply what TES could not cover
            % (do NOT subtract Q_solar again — it was already subtracted in STC_model)
            Q_residual(i) = max(Q_residual_in(i) - Q_TES_out(i), 0);
        end

        SOC_now = E_new / E_max;
        SOC(i)  = SOC_now;
    end

    fprintf('[TES]  Tank volume: %.0f m3\n', params.V_TES_m3);
    fprintf('[TES]  Annual energy delivered to process: %.1f MWh\n', sum(Q_TES_out)/1e6);
    fprintf('[TES]  Average annual SOC: %.1f%%\n', mean(SOC)*100);
end