%% params_system.m
% Global parameters for the hybrid solar thermal system.
% All physical values and design choices are centralised here.
% If anything needs to change, edit ONLY this file.
% =========================================================

%% --- INDUSTRY DATA (food factory, Seville, Spain) ---
P.load_W        = 10e6;         % [W]   Total thermal load = 10 MW
P.T_supply      = 100;          % [°C]  Supply temperature required by process
P.T_return      = 50;           % [°C]  Return temperature from process
P.hour_start    = 8;            % [h]   Production shift start (8:00 AM)
P.hour_end      = 18;           % [h]   Production shift end  (6:00 PM)

%% --- PROJECT TARGET ---
P.thermal_share = 0.60;         % [-]   60% of heat must come from STC + TES

%% --- SOLAR COLLECTORS: ABSOLICON T160 ---
% Data source: official Absolicon T160 datasheet
% Type: concentrating parabolic trough — uses DNI (direct normal irradiance)
P.eta0          = 0.762;        % [-]         Peak optical efficiency
P.a1            = 0.665;        % [W/(m²·K)]  Linear heat loss coefficient
P.a2            = 0.00378;      % [W/(m²·K²)] Quadratic heat loss coefficient
P.aperture_W    = 1.7;          % [m]    Aperture width of one collector
P.module_L      = 12.0;         % [m]    Length of one module
P.A_module      = P.aperture_W * P.module_L;  % [m²] Aperture area per module = 20.4 m²
P.DNI_min       = 50;           % [W/m²] Minimum DNI to operate the system

%% --- SOLAR FIELD SIZING ---
% The number of modules is calculated automatically in main_simulation.m
% based on real PVGIS data to meet the 60% thermal share target.
% This is just a starting guess that will be overwritten:
P.N_modules     = 250;          % [-]  Number of modules (initial guess)

%% --- TES: HOT WATER STORAGE TANK ---
% Choice: pressurised water, cost-effective and suitable up to ~120°C
% Source: standard industrial thermal storage literature
P.cp_water      = 4186;         % [J/(kg·K)] Specific heat capacity of water
P.rho_water     = 975;          % [kg/m³]    Density of water at ~75°C
P.T_TES_max     = 95;           % [°C]  Maximum temperature in the tank
P.T_TES_min     = 55;           % [°C]  Minimum useful temperature in the tank
P.deltaT_TES    = P.T_TES_max - P.T_TES_min;  % [°C] = 40°C useful swing

% TES capacity: must cover ~4 hours of solar load
% (early morning and late afternoon when sun is insufficient)
P.E_TES_Wh      = P.thermal_share * P.load_W * 4;  % [Wh] = 24 MWh
P.E_TES_J       = P.E_TES_Wh * 3600;               % [J]

% Required water volume: E = m·cp·deltaT → V = E / (rho·cp·deltaT)
P.V_TES_m3      = P.E_TES_J / (P.rho_water * P.cp_water * P.deltaT_TES);
% ≈ 520 m³ (roughly a tank 10m diameter × 7m height)

P.TES_loss_frac = 0.003;        % [1/h]  Thermal loss = 0.3% of stored energy/hour
P.SOC_max       = 1.00;         % [-]    Maximum state of charge (100%)
P.SOC_min       = 0.05;         % [-]    Minimum state of charge (5%, pump safety)
P.SOC_initial   = 0.50;         % [-]    Initial state of charge at start of year