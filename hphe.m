%% ========================================================
%  Heat Pump & Heat Engine Modelling
%  Seville, Spain
%  Inputs: ST hourly output from ST module
%% ========================================================

clc; clear; close all;

%% ── 1. PARAMETERS 系统参数 ───────────────────────────────

% Temperature settings 温度设定
T_hot_HP  = 157 + 273.15;   % Heat pump output temp [K] 热泵出口温度
T_cold_HP = 70  + 273.15;   % Heat pump input (waste heat) [K] 废热温度
T_hot_HE  = 157 + 273.15;   % Heat engine hot side [K] 热机高温侧
T_cold_HE = 30  + 273.15;   % Heat engine cold side (ambient) [K] 冷侧

% Carnot efficiency correction factor
% (real systems achieve ~40-50% of Carnot)
epsilon_HP = 0.5;           % Heat pump Carnot correction 热泵修正系数
epsilon_HE = 0.2;           % Heat engine Carnot correction 热机修正系数

% Industrial demand 工业热需求
P_demand   = 10e6;           % need ST peak thermal power
t_op       = 10;             % Operating hours per day [h]

% Waste heat availability 废热参数
Q_waste_max = 5e6;           % Max available waste heat [W] = 5 MW
                             % (adjust based on project spec)

%% ── 2. CALCULATE COP AND EFFICIENCY ─────────────────────

% Heat Pump COP
COP_carnot = T_hot_HP / (T_hot_HP - T_cold_HP);
COP_real   = epsilon_HP * COP_carnot;

% Heat Engine efficiency
eta_carnot = 1 - T_cold_HE / T_hot_HE;
eta_real   = epsilon_HE * eta_carnot;

fprintf('=== Theoretical Performance ===\n');
fprintf('HP Carnot COP:       %.2f\n', COP_carnot);
fprintf('HP Real COP:         %.2f\n', COP_real);
fprintf('Engine Carnot eta:   %.2f%%\n', eta_carnot*100);
fprintf('Engine Real eta:     %.2f%%\n', eta_real*100);
fprintf('================================\n\n');

%% ── 3. LOAD ST OUTPUT 读取ST模块输出 ─────────────────────
%
% Q_ST_hourly should come from your ST simulation (8760 values)
% If not available yet, generate a simplified estimate:

% Option A: Load from ST module output
% load('ST_output.mat', 'Q_hourly');
% Q_ST_hourly = Q_hourly;   % [Wh per hour]

% Option B: Simplified sinusoidal estimate (placeholder)
% Replace this with real ST output when available
t_vec = (1:8760)';
Q_ST_peak = 10.1e6;         % Peak ST output from ST sizing [W]
Q_ST_hourly = zeros(8760,1);
for i = 1:8760
    hour_of_day = mod(i-1, 24);
    day_of_year = floor((i-1)/24) + 1;
    % Solar availability: 7am-6pm, seasonal variation
    if hour_of_day >= 7 && hour_of_day <= 18
        solar_hour  = sin(pi*(hour_of_day-7)/11);
        seasonal    = 0.7 + 0.3*sin(2*pi*(day_of_year-80)/365);
        Q_ST_hourly(i) = Q_ST_peak * solar_hour * seasonal;
    end
end
% Convert to Wh (already per-hour values)

% Build hourly demand profile (only during operating hours)
Q_demand_hourly = zeros(8760, 1);
for i = 1:8760
    hour_of_day = mod(i-1, 24);
    % Operating schedule: 8am-6pm (adjust to your project)
    if hour_of_day >= 8 && hour_of_day <= 18
        Q_demand_hourly(i) = P_demand;   % [W]
    end
end

%% ── 4. HOURLY SIMULATION 逐时仿真 ────────────────────────

% Output arrays
Q_HP_hourly     = zeros(8760, 1);   % Heat pump heat output [Wh]
W_elec_hourly   = zeros(8760, 1);   % Heat pump electricity input [Wh]
Q_waste_hourly  = zeros(8760, 1);   % Waste heat consumed by HP [Wh]
W_engine_hourly = zeros(8760, 1);   % Heat engine electricity output [Wh]
Q_excess_hourly = zeros(8760, 1);   % Excess ST heat to engine [Wh]
Q_reject_hourly = zeros(8760, 1);   % Engine heat rejection [Wh]

for t = 1:8760

    Q_ST  = Q_ST_hourly(t);       % ST output this hour [Wh]
    Q_dem = Q_demand_hourly(t);   % Demand this hour [Wh]

    if Q_dem == 0
        % No demand (outside operating hours)
        continue
    end

    if Q_ST < Q_dem
        %% ── HEAT PUMP MODE 热泵模式 ──────────────────────
        % ST not enough → heat pump covers the gap

        Q_gap = Q_dem - Q_ST;           % Heat gap to fill [Wh]

        % Check waste heat availability
        Q_waste_avail = min(Q_waste_max, Q_gap * (1 - 1/COP_real));

        % Heat pump output
        Q_HP = COP_real * (Q_gap / COP_real) * COP_real;
        % Simplified: HP needs to produce Q_gap
        Q_HP = Q_gap;

        % Electricity consumed by heat pump
        W_elec = Q_HP / COP_real;

        % Waste heat consumed
        Q_waste_used = Q_HP - W_elec;

        % Store results
        Q_HP_hourly(t)    = Q_HP;
        W_elec_hourly(t)  = W_elec;
        Q_waste_hourly(t) = Q_waste_used;

    else
        %% ── HEAT ENGINE MODE 热机模式 ────────────────────
        % ST excess → heat engine generates electricity

        Q_excess = Q_ST - Q_dem;        % Excess heat [Wh]

        % Engine electricity output
        W_out    = eta_real * Q_excess;

        % Heat rejected to cold side
        Q_reject = Q_excess - W_out;

        % Store results
        Q_excess_hourly(t) = Q_excess;
        W_engine_hourly(t) = W_out;
        Q_reject_hourly(t) = Q_reject;
    end
end

%% ── 5. ANNUAL RESULTS 年度结果 ───────────────────────────

% Heat Pump results
E_HP_heat_MWh  = sum(Q_HP_hourly)    / 1e6;   % Annual heat from HP [MWh]
E_HP_elec_MWh  = sum(W_elec_hourly)  / 1e6;   % Annual electricity to HP [MWh]
E_waste_MWh    = sum(Q_waste_hourly) / 1e6;   % Annual waste heat used [MWh]

% Heat Engine results
E_engine_MWh   = sum(W_engine_hourly) / 1e6;  % Annual electricity generated [MWh]
E_excess_MWh   = sum(Q_excess_hourly) / 1e6;  % Annual excess ST heat [MWh]
E_reject_MWh   = sum(Q_reject_hourly) / 1e6;  % Annual heat rejected [MWh]

% Hours of operation
HP_hours     = sum(Q_HP_hourly > 0);
engine_hours = sum(W_engine_hourly > 0);

fprintf('=== Annual Results ===\n\n');
fprintf('-- Heat Pump --\n');
fprintf('Heat output:          %.1f MWh/year\n', E_HP_heat_MWh);
fprintf('Electricity input:    %.1f MWh/year\n', E_HP_elec_MWh);
fprintf('Waste heat used:      %.1f MWh/year\n', E_waste_MWh);
fprintf('Operating hours:      %d h/year\n',     HP_hours);
fprintf('Effective COP:        %.2f\n\n',         E_HP_heat_MWh/E_HP_elec_MWh);

fprintf('-- Heat Engine --\n');
fprintf('Electricity generated: %.1f MWh/year\n', E_engine_MWh);
fprintf('Excess heat input:     %.1f MWh/year\n', E_excess_MWh);
fprintf('Heat rejected:         %.1f MWh/year\n', E_reject_MWh);
fprintf('Operating hours:       %d h/year\n',     engine_hours);
fprintf('======================\n');

%% ── 6. MONTHLY BREAKDOWN 逐月分布 ────────────────────────

days_per_month  = [31,28,31,30,31,30,31,31,30,31,30,31];
hours_per_month = days_per_month * 24;
E_HP_monthly     = zeros(12,1);
E_engine_monthly = zeros(12,1);
h = 1;
for m = 1:12
    idx = h : h + hours_per_month(m) - 1;
    E_HP_monthly(m)     = sum(Q_HP_hourly(idx))     / 1e6;
    E_engine_monthly(m) = sum(W_engine_hourly(idx)) / 1e6;
    h = h + hours_per_month(m);
end

%% ── 7. PLOTS 画图 ────────────────────────────────────────

months = {'Jan','Feb','Mar','Apr','May','Jun',...
          'Jul','Aug','Sep','Oct','Nov','Dec'};

% Plot 1: Monthly HP heat output vs Engine electricity output
figure('Name','Monthly HP and Engine Output');
bar(1:12, [E_HP_monthly, E_engine_monthly]);
legend('HP Heat Output (MWh)', 'Engine Elec Output (MWh)');
xlabel('Month'); ylabel('Energy (MWh)');
title('Monthly Heat Pump & Engine Output - Seville');
xticklabels(months); grid on;

% Plot 2: COP sensitivity to temperature
figure('Name','COP vs Temperature');
T_cold_range = 50:5:100;
COP_curve    = epsilon_HP * T_hot_HP ./ (T_hot_HP - (T_cold_range+273.15));
plot(T_cold_range, COP_curve, 'b-o', 'LineWidth', 1.5);
xlabel('Waste Heat Temperature T_{cold} (°C)');
ylabel('Real COP');
title('Heat Pump COP vs Waste Heat Temperature');
grid on;

% Plot 3: Sample week hourly operation
figure('Name','Sample Week Operation');
t_sample = 1:168;   % first week
subplot(2,1,1);
area(t_sample, Q_ST_hourly(t_sample)/1e3, 'FaceColor',[1 0.8 0.2]);
hold on;
area(t_sample, Q_HP_hourly(t_sample)/1e3, 'FaceColor',[0.2 0.6 1]);
plot(t_sample, Q_demand_hourly(t_sample)/1e3, 'r--','LineWidth',1.5);
legend('ST Output (kWh)','HP Output (kWh)','Demand (kWh)');
xlabel('Hour'); ylabel('Energy (kWh)');
title('Heat Supply vs Demand - First Week');
grid on;

subplot(2,1,2);
bar(t_sample, W_engine_hourly(t_sample)/1e3, 'FaceColor',[0.2 0.8 0.4]);
xlabel('Hour'); ylabel('Electricity (kWh)');
title('Engine Electricity Output - First Week');
grid on;
