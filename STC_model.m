function [Q_solar, Q_residual, eta] = STC_model(Q_residual_in, climate, params)
%% STC_model.m
% Calculates the thermal power output of the Absolicon T160 solar collector
% field over a full year (8760 hours).
%
% INPUT:
%   Q_residual_in : [W] 8760×1 — Thermal demand to be met hour by hour
%   climate       : struct from load_climate_data.m (DNI, T_amb, etc.)
%   params        : struct from params_system.m (P)
%
% OUTPUT:
%   Q_solar    : [W] 8760×1 — Thermal power produced by the collectors
%   Q_residual : [W] 8760×1 — Remaining demand after STC contribution
%   eta        : [-] 8760×1 — Instantaneous collector efficiency
%
% PHYSICS — Efficiency formula (standard EN ISO 9806):
%   η = η0 − a1·(Tm−Ta)/G − a2·(Tm−Ta)²/G
%   where Tm = mean fluid temperature, Ta = ambient temperature, G = DNI
%   Thermal power: Q = η · A_total · G

    n = length(Q_residual_in);

    Q_solar    = zeros(n, 1);
    Q_residual = zeros(n, 1);
    eta        = zeros(n, 1);

    % Total aperture area of the solar field [m²]
    A_total = params.N_modules * params.A_module;

    % Mean fluid temperature inside the collectors [°C]
    T_m = (params.T_supply + params.T_return) / 2;  % = 75°C

    for i = 1:n
        G  = climate.DNI(i);     % [W/m²] Real DNI at hour i
        Ta = climate.T_amb(i);   % [°C]   Real ambient temperature at hour i

        if G > params.DNI_min
            % Reduced temperature difference (EN ISO 9806 standard parameter)
            X = (T_m - Ta) / G;  % [m²·K/W]

            % Collector efficiency
            eta_i = params.eta0 - params.a1 * X - params.a2 * X^2 * G;
            eta_i = max(eta_i, 0);  % Efficiency cannot be negative

            % Thermal power output [W]
            Q_i = eta_i * A_total * G;
        else
            % Night or overcast: no production
            eta_i = 0;
            Q_i   = 0;
        end

        eta(i)     = eta_i;
        Q_solar(i) = Q_i;

        % Residual demand: STC covers up to the current demand, no more
        Q_covered_i   = min(Q_i, Q_residual_in(i));
        Q_residual(i) = Q_residual_in(i) - Q_covered_i;
    end

    % Annual summary
    fprintf('[STC]  Modules: %d  |  Field area: %.0f m²\n', ...
            params.N_modules, A_total);
    fprintf('[STC]  Annual energy produced: %.1f MWh\n', sum(Q_solar)/1e6);
    fprintf('[STC]  Average efficiency (sun hours): %.1f%%\n', ...
            mean(eta(eta > 0)) * 100);
end