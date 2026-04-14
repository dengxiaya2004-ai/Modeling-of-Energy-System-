%% main_simulation.m
% =========================================================
% MAIN SCRIPT — Hybrid solar thermal system simulation
% Food factory, Seville, Spain
% Annual simulation: 8760 hours
%
% HOW THE TEAM SHOULD USE THIS FILE:
%   Each team member adds their own component function in the
%   CASCADE section below (Step 4). The only rule is:
%   your function receives Q_residual as input and returns
%   your Q_out and the reduced Q_residual as output.
%
% FILE STRUCTURE:
%   params_system.m       -> all physical parameters        (your file)
%   load_climate_data.m   -> loads real PVGIS data          (your file)
%   STC_model.m           -> solar collector model          (your file)
%   TES_model.m           -> thermal storage model          (your file)
%   heat_pump_model.m     -> heat pump model                (teammate)
%   el_heater_model.m     -> electric heater model          (teammate)
% =========================================================
clear; clc; close all;

%% -------------------------------------------------------
%% STEP 1: LOAD PARAMETERS AND CLIMATE DATA
%% -------------------------------------------------------
params_system;   % Runs params file, creates struct P with all parameters
fprintf('=== Hybrid solar thermal system — Seville, Spain ===\n\n');

% Load real hourly data from PVGIS CSV file (must be in the project folder)
climate = load_climate_data('seville_tmy.csv');

%% -------------------------------------------------------
%% STEP 2: ANNUAL THERMAL DEMAND PROFILE
%% -------------------------------------------------------
% The factory requires 10 MW of heat from 8:00 to 18:00, every day
n = 8760;
Q_demand = zeros(n, 1);

for i = 1:n
    h = climate.hour_of_day(i);  % hour of the day (0..23)
    if h >= P.hour_start && h < P.hour_end
        Q_demand(i) = P.load_W;  % 10 MW during production shifts
    end
end

fprintf('[Demand] Total annual energy demand:  %.1f MWh\n', sum(Q_demand)/1e6);
fprintf('[Demand] Annual operating hours:      %d h\n\n',   sum(Q_demand > 0));

%% -------------------------------------------------------
%% STEP 3: AUTOMATIC SOLAR FIELD SIZING
%% -------------------------------------------------------
% Find the number of modules so that STC + TES covers exactly 60%
% of annual demand. Uses a simple iterative search (50 to 1000 modules).

fprintf('[Sizing] Optimising number of STC modules...\n');

E_target_MWh = P.thermal_share * sum(Q_demand) / 1e6;
fprintf('[Sizing] Target energy from STC+TES: %.1f MWh/year\n', E_target_MWh);

best_modules = P.N_modules;
best_error   = Inf;

for n_mod = 50:10:1000
    P_test           = P;
    P_test.N_modules = n_mod;

    [Q_s_test, Q_res_test, ~] = STC_model(Q_demand, climate, P_test);
    [Q_t_test, ~, ~] = TES_model(Q_s_test, Q_demand, Q_res_test, P_test);

    Q_covered_test = min(Q_s_test, Q_demand) + Q_t_test;
    E_covered_test = sum(Q_covered_test) / 1e6;

    err = abs(E_covered_test - E_target_MWh);
    if err < best_error
        best_error   = err;
        best_modules = n_mod;
    end
end

P.N_modules = best_modules;
fprintf('[Sizing] Optimal number of modules: %d (field area: %.0f m²)\n\n', ...
        P.N_modules, P.N_modules * P.A_module);

%% -------------------------------------------------------
%% STEP 4: CASCADE SIMULATION
%% -------------------------------------------------------
% Q_residual is passed from one component to the next.
% Each component covers what it can and returns the reduced residual.
% ---------------------------------------------------------------
% ADD YOUR COMPONENT HERE following this template:
%
%   [Q_yourcomp, Q_residual] = yourcomp_model(Q_residual, climate, P);
%
% The variable Q_residual is automatically updated at each step.
% ---------------------------------------------------------------

fprintf('=== RUNNING ANNUAL SIMULATION ===\n');

% --- YOUR COMPONENTS (STC + TES) ---
[Q_solar, Q_residual, eta_solar] = STC_model(Q_demand, climate, P);
[Q_TES,   Q_residual, SOC_TES]   = TES_model(Q_solar, Q_demand, Q_residual, P);

% --- TEAMMATES' COMPONENTS ---
% Uncomment and complete these lines once teammates share their functions:
%
% [Q_HP,     Q_residual] = heat_pump_model(Q_residual, climate, P);
% [Q_boiler, Q_residual] = el_heater_model(Q_residual, climate, P);

% Whatever is left in Q_residual at this point needs to be covered by backup
Q_backup = Q_residual;

%% -------------------------------------------------------
%% STEP 5: RESULTS AND THERMAL SHARE VERIFICATION
%% -------------------------------------------------------
E_demand_MWh  = sum(Q_demand)               / 1e6;
E_solar_MWh   = sum(min(Q_solar, Q_demand)) / 1e6;
E_TES_MWh     = sum(Q_TES)                  / 1e6;
E_backup_MWh  = sum(Q_backup)               / 1e6;
E_covered_MWh = E_solar_MWh + E_TES_MWh;

thermal_share_real = E_covered_MWh / E_demand_MWh * 100;

fprintf('\n============= ANNUAL RESULTS =============\n');
fprintf('Total demand:                %8.1f MWh\n',           E_demand_MWh);
fprintf('Energy from STC (direct):    %8.1f MWh  (%.1f%%)\n', ...
        E_solar_MWh, E_solar_MWh/E_demand_MWh*100);
fprintf('Energy from TES:             %8.1f MWh  (%.1f%%)\n', ...
        E_TES_MWh,   E_TES_MWh/E_demand_MWh*100);
fprintf('Total STC + TES:             %8.1f MWh  (%.1f%%)\n', ...
        E_covered_MWh, thermal_share_real);
fprintf('Backup required (HP+boiler): %8.1f MWh  (%.1f%%)\n', ...
        E_backup_MWh, E_backup_MWh/E_demand_MWh*100);
fprintf('------------------------------------------\n');
fprintf('Target thermal share:        %.0f%%\n',   P.thermal_share*100);
fprintf('Achieved thermal share:      %.1f%%\n',   thermal_share_real);
fprintf('==========================================\n\n');

%% -------------------------------------------------------
%% STEP 6: SAVE SHARED OUTPUT FOR THE TEAM
%% -------------------------------------------------------
% Teammates load this file with: load('system_results.mat')
results.hours        = climate.hours;
results.month        = climate.month;
results.hour_of_day  = climate.hour_of_day;
results.Q_demand_W   = Q_demand;
results.Q_solar_W    = Q_solar;
results.Q_TES_W      = Q_TES;
results.Q_backup_W   = Q_backup;
results.SOC_TES      = SOC_TES;
results.DNI          = climate.DNI;
results.T_amb        = climate.T_amb;
results.P            = P;

save('system_results.mat', 'results');
fprintf('Results saved to system_results.mat\n');

%% -------------------------------------------------------
%% STEP 7: PLOTS
%% -------------------------------------------------------
months_label = {'Jan','Feb','Mar','Apr','May','Jun', ...
                'Jul','Aug','Sep','Oct','Nov','Dec'};

% --- Plot 1: Monthly aggregated energy breakdown ---
figure('Name', 'Monthly energy breakdown', 'Position', [50 50 900 480]);
E_monthly = zeros(12, 3);
for m = 1:12
    idx = (climate.month == m);
    E_monthly(m,1) = sum(min(Q_solar(idx), Q_demand(idx))) / 1e6;
    E_monthly(m,2) = sum(Q_TES(idx))    / 1e6;
    E_monthly(m,3) = sum(Q_backup(idx)) / 1e6;
end
bar(E_monthly, 'stacked');
colororder({'#F5A623','#4A90D9','#D0021B'});
set(gca, 'XTickLabel', months_label);
ylabel('Thermal energy [MWh]');
title('Monthly thermal energy breakdown — STC + TES + Backup');
legend('STC (direct)','TES (storage)','Backup required','Location','north');
grid on;

% --- Plot 2: TES state of charge — typical summer week ---
figure('Name', 'TES SOC summer week', 'Position', [50 580 900 340]);
idx_june_week = find(climate.month == 6, 168, 'first');
plot(1:168, SOC_TES(idx_june_week)*100, 'b-', 'LineWidth', 1.5);
xlabel('Hour of the week');
ylabel('SOC [%]');
title('TES state of charge — typical summer week (June)');
yline(P.SOC_min*100, 'r--', 'Min SOC');
yline(100, 'g--', 'Max SOC');
ylim([0 110]); grid on;
xticks(0:24:168);
xticklabels({'Mon','Tue','Wed','Thu','Fri','Sat','Sun',''});

% --- Plot 3: Hourly power balance — typical winter week ---
figure('Name', 'Hourly power balance winter week', 'Position', [50 340 900 380]);
idx_jan_week = find(climate.month == 1, 168, 'first');
hold on;
area(1:168, Q_demand(idx_jan_week)/1e6, ...
     'FaceColor','#E8E8E8','EdgeColor','none');
area(1:168, min(Q_solar(idx_jan_week), Q_demand(idx_jan_week))/1e6, ...
     'FaceColor','#F5A623','FaceAlpha',0.85,'EdgeColor','none');
area(1:168, Q_TES(idx_jan_week)/1e6, ...
     'FaceColor','#4A90D9','FaceAlpha',0.85,'EdgeColor','none');
plot(1:168, Q_demand(idx_jan_week)/1e6, 'k--', 'LineWidth', 1.5);
xlabel('Hour of the week');
ylabel('Thermal power [MW]');
title('Hourly thermal power balance — typical winter week (January)');
legend('Backup','STC','TES','Demand','Location','northwest');
xticks(0:24:168);
xticklabels({'Mon','Tue','Wed','Thu','Fri','Sat','Sun',''});
grid on;