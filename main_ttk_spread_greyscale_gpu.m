%% main_ttk_spread_greyscale_gpu.m
% Two 3D plots (q,h,dist):
%   1) Δ12 = TTK_rank2 - TTK_rank1
%   2) Δ23 = TTK_rank3 - TTK_rank2
%
% GREY RULE (as you described):
%   greyness = min(Δt / 1.0, 1)
%   Δt = 0.0  -> white
%   Δt = 0.4  -> 40% towards black
%   Δt >= 1.0 -> black
%
% GPU compute + waitbar progress.

clear; clc;

%% ---------- SETTINGS ----------
useGPU = true;

% Grid (keep same style as before)
q_vals    = linspace(0.10, 1.00, 46);
h_vals    = linspace(0.00, 1.00, 46);
dist_vals = linspace(0, 50, 51);

markerSize = 10;

% Your greyness scale: 1 second = fully black
greyFullSec = 1.0;

%% ---------- FIND GUN FUNCTIONS ----------
files = dir('ttk_*.m');
names = strings(0,1);
for k = 1:numel(files)
    [~, fn] = fileparts(files(k).name);
    if startsWith(fn,"ttk_")
        names(end+1,1) = string(fn);
    end
end
names = unique(sort(names));
G = numel(names);
if G < 3
    error('Need at least 3 gun functions named ttk_*.m. Found %d.', G);
end

fprintf('Found %d guns:\n', G);
disp(names);

%% ---------- BUILD GRID (CPU for plotting) ----------
[Qcpu, Hcpu, Dcpu] = ndgrid(q_vals, h_vals, dist_vals);
sz = size(Qcpu);
N  = numel(Qcpu);

% One-time validation on CPU
if any(Qcpu(:) < 0 | Qcpu(:) > 1), error('q must be in [0,1]'); end
if any(Hcpu(:) < 0 | Hcpu(:) > 1), error('h must be in [0,1]'); end
if any(Dcpu(:) < 0),               error('dist_m must be >= 0'); end

%% ---------- GPU SETUP ----------
if useGPU
    try
        dev = gpuDevice;
        fprintf('Using GPU: %s | TotalMemory %.2f GB\n', dev.Name, dev.TotalMemory/1e9);
        Q = gpuArray(single(Qcpu));
        H = gpuArray(single(Hcpu));
        D = gpuArray(single(Dcpu));
    catch ME
        warning('GPU unavailable (%s). Falling back to CPU.', ME.message);
        useGPU = false;
        Q = single(Qcpu); H = single(Hcpu); D = single(Dcpu);
    end
else
    Q = single(Qcpu); H = single(Hcpu); D = single(Dcpu);
end

%% ---------- ALLOCATE TTK ----------
if useGPU
    TTK = gpuArray.nan(N, G, 'single');
else
    TTK = nan(N, G, 'single');
end

%% ---------- PROGRESS ----------
wb = waitbar(0,'Starting...','Name','Evaluating guns');
t0 = tic;

for g = 1:G
    f = str2func(names(g));

    Ettk = f(Q,H,D);                  % Ettk only
    TTK(:,g) = reshape(Ettk,[N 1]);

    frac = g/G;
    elapsed = toc(t0);
    eta = elapsed*(1/frac - 1);

    if useGPU
        dev = gpuDevice;
        usedGB = (dev.TotalMemory - dev.AvailableMemory)/1e9;
        msg = sprintf('Gun %d/%d: %s\nElapsed %.1fs | ETA %.1fs\nGPU mem %.2f / %.2f GB', ...
            g, G, names(g), elapsed, eta, usedGB, dev.TotalMemory/1e9);
    else
        msg = sprintf('Gun %d/%d: %s\nElapsed %.1fs | ETA %.1fs', ...
            g, G, names(g), elapsed, eta);
    end

    waitbar(frac, wb, msg);
end
if ishandle(wb), delete(wb); end

%% ---------- TOP-3 TTK VALUES (NO SORT) ----------
[ttk1, idx1] = min(TTK, [], 2);

lin = (1:N).';
if useGPU, lin = gpuArray(lin); end

TTK2 = TTK;
TTK2(sub2ind([N,G], lin, idx1)) = inf;
[ttk2, idx2] = min(TTK2, [], 2);

TTK3 = TTK2;
TTK3(sub2ind([N,G], lin, idx2)) = inf;
[ttk3, ~] = min(TTK3, [], 2);

% Gather TTKs to CPU
ttk1 = gather(ttk1);
ttk2 = gather(ttk2);
ttk3 = gather(ttk3);

%% ---------- DIFFERENCES ----------
d12 = ttk2 - ttk1;          % rank2 - rank1
d23 = ttk3 - ttk2;          % rank3 - rank2

% Any Inf/NaN gaps => treat as huge gap (black)
d12(~isfinite(d12)) = greyFullSec;
d23(~isfinite(d23)) = greyFullSec;

% Clamp to >=0
d12 = max(d12, 0);
d23 = max(d23, 0);

%% ---------- MAP TO GREYSCALE RGB (your rule) ----------
% greyness fraction: 0 -> white, 1 -> black
g12 = min(d12 ./ greyFullSec, 1);   % 0..1
g23 = min(d23 ./ greyFullSec, 1);   % 0..1

% Convert to RGB: white(1) -> black(0)
% If g=0 => color=1 (white). If g=1 => color=0 (black).
rgb12 = repmat(1 - g12, 1, 3);
rgb23 = repmat(1 - g23, 1, 3);

%% ---------- PLOTTING ----------
Qp = Qcpu(:); Hp = Hcpu(:); Dp = Dcpu(:);

plotGrey3D(Qp,Hp,Dp,rgb12,markerSize, ...
    sprintf('ΔTTK = Rank#2 - Rank#1 (shade: 0s white → 1s black) | grid=%dx%dx%d', sz(1),sz(2),sz(3)));

plotGrey3D(Qp,Hp,Dp,rgb23,markerSize, ...
    sprintf('ΔTTK = Rank#3 - Rank#2 (shade: 0s white → 1s black) | grid=%dx%dx%d', sz(1),sz(2),sz(3)));

%% ---------- LOCAL ----------
function plotGrey3D(q,h,d,rgb,ms,titleTxt)
    figure('Color','w');
    scatter3(q,h,d,ms,rgb,'filled', ...
        'MarkerFaceAlpha',0.9,'MarkerEdgeAlpha',0.9);

    grid on;
    xlabel('Accuracy q');
    ylabel('Headshot fraction h');
    zlabel('Distance (m)');
    title(titleTxt,'Interpreter','none');
    view(3);

    % Manual grayscale legend (1s scale)
    colormap(flipud(gray(256)));
    cb = colorbar;
    caxis([0 1]);
    ylabel(cb,'Greyness fraction (Δt / 1s)');
end
