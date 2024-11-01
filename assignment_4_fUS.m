%%%%%%%%%%%%%%%%%% Clear previous work and add paths %%%%%%%%%%%%%%%%%%
close all; 
clear; 
clc; 

addpath(genpath('given'));
addpath(genpath('data'));

%%%%%%%%%%%%%%%% Load experimental data and set parameters %%%%%%%%%%%%%%%%
load('params.mat', 'params'); 
x_axis = params.x_axis; % Pixel locations along width [mm]
z_axis = params.z_axis; % Pixel locations along depth [mm]
Fs = params.Fs; % Sampling rate of the experiment

% Load the power-Doppler images
load('pdi.mat', 'PDI'); 

% Load the binary stimulus vector
load('stim.mat', 'stim');

Nz = size(PDI, 1); % Number of pixels along the depth dimension
Nx = size(PDI, 2); % Number of pixels along the width dimension
Nt = size(PDI, 3); % Number of timestamps
t_axis = 0 : 1 / Fs : (Nt - 1) / Fs; % Time axis of the experiment

%%%%%%%%%%%%%%%%%% Get to know the data %%%%%%%%%%%%%%%%%%
% Choose a timestamp and show the PDI for this timestamp
idx_t = 100;       
%
figure;
imagesc(x_axis, z_axis, PDI(:, :, idx_t)); 
xlabel('Width [mm]');
ylabel('Depth [mm]'); 
title(['PDI at ' num2str(t_axis(100)) ' seconds']);

% Choose a pixel and plot the time series for this pixel
idx_z = 85;
idx_x = 30;
pxl_time_series = squeeze(PDI(idx_z, idx_x, :));
offset = min(pxl_time_series); 
wid = max(pxl_time_series) - min(pxl_time_series);
%
figure; 
plot(t_axis, pxl_time_series); 
hold on; 
s = plot(t_axis, offset + wid * stim); % the stimulus is offset and multiplied 
        % to visualize it at the same y-axis scale as the fUS time series
xlabel('Time [s]'); 
ylabel('Power Doppler amplitude [-]');
title(['Time series of the pixel at (' num2str(z_axis(10)) 'mm, ' num2str(x_axis(10)) 'mm)']); 
legend(s, 'Stimulus');

% Show the mean PDI
% Calculate the mean PDI
mean_PDI = mean(PDI, 3);
mean_PDI = mean_PDI./(max(mean_PDI(:)));

% Display the log of mean_PDI to enhance the contrast
figure; 
imagesc(x_axis, z_axis, log(mean_PDI));  
title('Mean PDI')
ylabel('Depth [mm]')
xlabel('Width [mm]')

%%%%%%%%%%%%%%%%%%%%% Data preprocessing %%%%%%%%%%%%%%%%%%%%%
% Standardize the time series for each pixel
P = (PDI - mean(PDI, 3)) ./ std(PDI, [], 3); 

% Spatial Gaussian smoothing
ht = fspecial('gaussian', [4 4], 2);
Pg = double(convn(P, ht, 'same'));

% Filter the pixel time-series with a temporal low pass filter at 0.3 Hz 
f1 = 0.3;
[b, a] = butter(5, f1 / (Fs / 2), 'low');
PDImatrix = reshape(Pg, Nz * Nx, Nt);
Pgf = reshape(filtfilt(b, a, PDImatrix')', size(PDI));
PDI = Pgf;
PDI_matrix = reshape(PDI, Nz * Nx, Nt);
clear P Pg Pgf
%%
%%%% Calculate the best correlation lag and show the correlation image %%%%
% Initialize variables
max_lag_seconds = 10;  % Maximum lag in seconds
lags = 0:1/Fs:max_lag_seconds;  % Lags from 0 to 10 seconds in steps of 1/Fs
mean_corr_values = zeros(size(lags));  % To store mean correlation values for each lag

% Loop over all lags
for i = 1:length(lags)
    % Shift the stimulus by current lag
    shift_samples = round(lags(i) * Fs);
    shifted_stim = [zeros(shift_samples, 1); stim(1:end-shift_samples)];  % Zero-padding shift
    
    % Calculate correlation for each pixel's time series with shifted stimulus
    corr_image = zeros(Nz, Nx);
    for z = 1:Nz
        for x = 1:Nx
            pixel_time_series = squeeze(PDI(z, x, :));  % Extract time series for the pixel
            corr_image(z, x) = abs(corr(pixel_time_series, shifted_stim));  % abs Pearson correlation
        end
    end
    
    % Store mean absolute correlation across all pixels
    mean_corr_values(i) = mean(abs(corr_image(:)));
end

% Find the optimal lag with the maximum mean correlation
[~, best_lag_idx] = max(mean_corr_values);
best_lag = lags(best_lag_idx);

% Shift the stimulus by the best lag and calculate the final correlation image
shift_samples = round(best_lag * Fs);
shifted_stim = [zeros(shift_samples, 1); stim(1:end-shift_samples)];
stim_best_lag = shifted_stim;
pc_image = zeros(Nz, Nx);
for z = 1:Nz
    for x = 1:Nx
        pixel_time_series = squeeze(PDI(z, x, :));
        pc_image(z, x) = corr(pixel_time_series, shifted_stim);  % Final correlation
    end
end
%%
% Two ways to visualize the correlation image are provided
plot_version = 1;
display_brain_img(pc_image, log(mean_PDI), z_axis, x_axis, ...
    'Significantly Correlated Regions', plot_version);

plot_version = 2;
display_brain_img(pc_image, log(mean_PDI), z_axis, x_axis, ...
    'Significantly Correlated Regions', plot_version);

%%
% A_init = randn(size(PDI, 1), R);
% B_init = randn(size(PDI, 2), R);
% C_init = randn(size(PDI, 3), R);
% set(gcf,'Units','inches');
% screenposition = get(gcf,'Position');
% set(gcf,...
%     'PaperPosition',[0 0 20 15],...
%     'PaperSize',[20 15]);
% print -dpdf -painters significantly_correlated_image

% width=6;
% height=5;
% set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperPosition', [0, 0, width, height]); % Remove margins
% set(gcf, 'PaperSize', [width, height]); % Set exact size for PDF output
% % Save current figure (gcf) as PDF
% print(gcf, 'Correlation_image.pdf', '-dpdf', '-vector'); % '-vector' for vector graphics
%%
%%%%%%%%%%%%%%%%%%%%%%%%%% CPD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can use hidden_cpd_als_3d.m for this part.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)

close all;
R = 3;
A_init = randn(size(PDI, 1), R);
B_init = randn(size(PDI, 2), R);
C_init = randn(size(PDI, 3), R);
options.maxiter = 100; 
options.th_relerr = 0.7930;
[B1, B2, B3, c, output_cpd] = hidden_cpd_als_3d(PDI, R, options, A_init, B_init, C_init);

cpd_corrs = abs(corr(B3, shifted_stim));
% figure;
for r = 1:R
    spatial_image = B1(:,r) * B2(:,r)';
    % Plot the spatial map
    figure;
    imagesc(x_axis, z_axis, spatial_image);
    xlabel('Width [mm]');
    ylabel('Depth [mm]');
    title(['Spatial Map for Component ', num2str(r)]);
    colorbar;
    set(gca, 'YDir', 'reverse'); % To display depth correctly
    
%     % Plot the first subplot
%     subplot(1, 3, r);
%     imagesc(x_axis, z_axis, abs(spatial_image));
%     title(['Component ', num2str(r)]);
%     set(gca, 'YDir', 'reverse'); % To display depth correctly
%     axis image;
    
%     % Plot the temporal factor
%     figure;
%     temporal_factor = B3(:, r);
%     plot(t_axis, temporal_factor);
%     hold on;
%     plot(t_axis, shifted_stim);
%     ylim([-0.2 0.2]);
%     legend('Temporal Signature', 'Shifted Stimulus');
%     xlabel('Time [s]');
%     ylabel('Amplitude');
%     title(['Temporal Signature for Component ', num2str(r)]);
end

% % Add a colorbar that applies to all plots
% h = colorbar;
% h.Position = [0.92 0.35 0.02 0.315]; % Adjust position and size of colorbar
%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BTD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fill in btd_ll1_als_3d.m.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)

start_idx = 1;
%for R = 1:4
R = 20;
Lr = 10*ones(1,R);
[A, B, C, const, output] = btd_ll1_als_3d(PDI, R, Lr, options);
   for r=1:R
        L = Lr(r);                 % Rank of component r
        end_idx = start_idx + L - 1;
        
        % Extract factor matrices for component r
        A_r = A(:, start_idx:end_idx);  % Size (Nz, L)
        B_r = B(:, start_idx:end_idx);  % Size (Nx, L)
        
        % Compute spatial map
        spatial_map_r = A_r * B_r';
        
        % Adjust the sign if necessary (due to sign ambiguity)
        sign_adjustment = sign(sum(spatial_map_r(:)));
        spatial_map_r = spatial_map_r * sign_adjustment;
        
        % Plot the spatial map
        figure;
        imagesc(x_axis, z_axis, spatial_map_r);
        xlabel('Width [mm]');
        ylabel('Depth [mm]');
        title(['Spatial Map for Component ', num2str(r)]);
        colorbar;
        set(gca, 'YDir', 'reverse'); % To display depth correctly
        filename = sprintf('temporal_%d_withR_%d_Lr_%d.eps', r,R,Lr(r));
        % Save the current figure as .eps
        saveas(gcf, filename, 'epsc'); % 'epsc' saves in color
        % Extract temporal factor for component r
        C_r = C(:, r);  % Assuming C is of size (Nt, R)
        temporal_factor = C_r * sign_adjustment;
        
        % Plot the temporal factor
        %figure;
        %plot(t_axis, temporal_factor);
        %xlabel('Time [s]');
        %ylabel('Amplitude');
        %title(['Temporal Signature for Component ', num2str(r)]);

        % Plot the temporal factor
        figure;
        plot(t_axis, temporal_factor);
        hold on;
        plot(t_axis, shifted_stim);
        ylim([-0.2 0.2]);
        legend('Temporal Signature', 'Shifted Stimulus');
        xlabel('Time [s]');
        ylabel('Amplitude');
        title(['Temporal Signature for Component ', num2str(r)]);
        filename = sprintf('corr_image_%d_withR_%d_Lr_%d.eps', r,R,Lr(r));
        % Save the current figure as .eps
        saveas(gcf, filename, 'epsc'); % 'epsc' saves in color
        
        % Compute correlation with the shifted stimulus
        corr_coeff = corr(temporal_factor, stim_best_lag);
        disp(['Correlation between temporal factor of component ', num2str(r), ' and shifted stimulus: ', num2str(corr_coeff)]);
        %filename = sprintf('corr_image_%d_withR_%d_Lr_%d.eps', r,R,Lr(r));
        % Save the current figure as .eps
        %saveas(gcf, filename, 'epsc'); % 'epsc' saves in color
        % Update start index for next component
        start_idx = end_idx + 1;
        
        %pause; % Pause to inspect each component
    end
%end


%%
%%asd
% Plot the relative error per iteration
figure;
semilogy(output.relerr)
grid on
xlabel('Iteration number')
ylabel('Relative error $\frac{\| T-T_{dec} \|_F}{\| T\|_F}$','interpreter','latex');
title('Relative error of the decomposition')


%%
% R=3;
% Lr=[1,1,1];
% options.maxiter = 300; 
% options.th_relerr = 1e-6;
% 
% % init matrices
% U1 = randn(size(PDI, 1), sum(Lr));
% U2 = randn(size(PDI, 2), sum(Lr));
% U3 = randn(size(PDI, 3), R);
% 
% [A, B, C, const, output] = btd_ll1_als_3d(PDI, R, Lr, options, U1, U2, U3);
% [B1, B2, B3, c, output2] = cpd_als_3d(PDI, R, options, U1, U2, U3);
% 
% %%
% output.relerr == output2.relerr
