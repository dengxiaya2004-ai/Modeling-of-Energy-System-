function climate = load_climate_data(filepath)
%% load_climate_data.m
% Loads real climate data from a PVGIS TMY (Typical Meteorological Year)
% CSV file downloaded manually from: https://re.jrc.ec.europa.eu/pvg_tools
%
% HOW TO GET THE FILE:
%   1. Go to re.jrc.ec.europa.eu/pvg_tools
%   2. Click "TMY" in the left menu
%   3. Search for "Seville, Spain" on the map
%   4. Click Download → select CSV format
%   5. Save the file as 'seville_tmy.csv' in your project folder
%
% INPUT:
%   filepath : string with the path to the CSV file
%              Example: 'seville_tmy.csv'
%
% OUTPUT:
%   climate  : struct with hourly data for 8760 hours (full year)
%     .DNI          [W/m²]   Direct Normal Irradiance (used by STC collectors)
%     .GHI          [W/m²]   Global Horizontal Irradiance
%     .T_amb        [°C]     Ambient air temperature at 2m height
%     .hours        [1..8760] Hour index of the year
%     .month        [1..12]   Month corresponding to each hour
%     .hour_of_day  [0..23]   Hour of the day corresponding to each hour

    %% STEP A: Check that the file exists
    if ~isfile(filepath)
        error(['File not found: %s\n\n' ...
               'Please download the TMY CSV file manually from PVGIS:\n' ...
               '  1. Go to: https://re.jrc.ec.europa.eu/pvg_tools\n' ...
               '  2. Click "TMY" in the left menu\n' ...
               '  3. Search for Seville, Spain on the map\n' ...
               '  4. Click Download -> CSV\n' ...
               '  5. Save the file as ''seville_tmy.csv'' in your project folder'], ...
               filepath);
    end

    fprintf('Loading climate data from: %s\n', filepath);

    %% STEP B: Find the header row inside the CSV
    % The PVGIS CSV file has this structure:
    %   Rows 1-16:    metadata (location, data source, etc.) — skip these
    %   Row ~17:      column names: "time(UTC),T2m,RH,Gb(n),Gd(h),..."
    %   Rows 18-8777: 8760 rows of hourly data
    %   Final rows:   footnotes — ignore these
    %
    % Strategy: search for the row containing "time(UTC)" to find
    % the exact start of data (robust even if the format changes slightly).

    fid = fopen(filepath, 'r');
    if fid == -1
        error('Could not open file: %s', filepath);
    end

    header_line = '';
    header_row  = 0;
    row_count   = 0;

    while ~feof(fid)
        line = fgetl(fid);
        row_count = row_count + 1;
        if contains(line, 'time(UTC)')
            header_line = line;
            header_row  = row_count;
            break;
        end
    end
    fclose(fid);

    if isempty(header_line)
        error(['Column header not found in the PVGIS file.\n' ...
               'Make sure you downloaded the correct CSV format from PVGIS TMY.']);
    end

    %% STEP C: Identify the columns we need
    col_names = strsplit(strtrim(header_line), ',');

    % Standard PVGIS column names
    idx_DNI  = find(strcmp(col_names, 'Gb(n)'));   % Direct Normal Irradiance
    idx_GHI  = find(strcmp(col_names, 'Gd(h)'));   % Diffuse Horizontal Irradiance
    idx_Tamb = find(strcmp(col_names, 'T2m'));      % Air temperature at 2m

    if isempty(idx_DNI) || isempty(idx_Tamb)
        error(['Required columns not found in the file.\n' ...
               'Available columns: %s\n' ...
               'Expected: Gb(n) for DNI, T2m for temperature.'], ...
               strjoin(col_names, ', '));
    end


%% STEP D: Read the 8760 hourly data rows
    opts = detectImportOptions(filepath, 'NumHeaderLines', header_row);
    opts.DataLines          = [header_row + 1, header_row + 8760];
    opts.VariableNames      = col_names;
    opts.VariableNamingRule = 'preserve';   % <-- THIS IS THE FIX
                                            % Keeps original names like Gb(n)
                                            % instead of letting MATLAB rename them
    opts = setvartype(opts, col_names(2:end), 'double');

    T = readtable(filepath, opts);
    
    %% STEP E: Extract columns and build time vectors
    n = 8760;  % Hours in one year (365 × 24)

    DNI_raw  = T.(col_names{idx_DNI})(1:n);
    GHI_raw  = T.(col_names{idx_GHI})(1:n);
    Tamb_raw = T.(col_names{idx_Tamb})(1:n);

    % Remove physically impossible negative values
    DNI_raw  = max(DNI_raw, 0);
    GHI_raw  = max(GHI_raw, 0);

    %% STEP F: Build month and hour-of-day vectors
    days_per_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    month_vec      = zeros(n, 1);
    hour_of_day    = zeros(n, 1);

    idx = 1;
    for m = 1:12
        for d = 1:days_per_month(m)
            for h = 0:23
                if idx > n; break; end
                month_vec(idx)   = m;
                hour_of_day(idx) = h;
                idx = idx + 1;
            end
        end
    end

    %% STEP G: Build output struct
    climate.DNI         = DNI_raw;
    climate.GHI         = GHI_raw;
    climate.T_amb       = Tamb_raw;
    climate.hours       = (1:n)';
    climate.month       = month_vec;
    climate.hour_of_day = hour_of_day;

    % Summary statistics
    fprintf('[Climate] Annual average DNI:           %.1f W/m²\n', mean(DNI_raw));
    fprintf('[Climate] Peak DNI:                     %.1f W/m²\n', max(DNI_raw));
    fprintf('[Climate] Annual average temperature:   %.1f °C\n',   mean(Tamb_raw));
    fprintf('[Climate] Hours with sun (DNI>50 W/m²): %d h/year\n', sum(DNI_raw > 50));
end