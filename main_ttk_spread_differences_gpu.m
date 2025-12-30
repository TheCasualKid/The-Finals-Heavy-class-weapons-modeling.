%% main_ttk_spread_differences_gpu.m
% NEW code: generates TWO 3D graphs (same axes: q, h, dist)
%   Plot A: (TTK rank#2 - rank#1)
%   Plot B: (TTK rank#3 - rank#2)
% Color rule:
%   0s difference  -> WHITE
%   >= 2s          -> BLACK
% Uses GPU (optional) + progress bar.

clear; clc;

%% ---------- SETTINGS ----------
useGPU = true;

% Grid (match your earlier grid)
q_vals    = linspace(0.10, 1.00, 46);
h_vals    = linspace(0.00, 1.00, 46);
dist_vals = linspace(0, 50, 51);

markerSize = 9;
diffCap = 2.0;  % 2 seconds or more -> black

%% ---------- FIND GUN FUNCTIONS ----------
files = dir('ttk_*.m');
names = strings(0,1);
for k = 1:numel(files)
    [~, fn] = fileparts(files(k).name);
    % skip obvious "main" scripts
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

% one-time validation
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

%% ---------- PROGRESS BAR ----------
wb = waitbar(0,'Starting...','Name','Evaluating guns');
setappdata(wb,'canceling',0);
t0 = tic;

%% ---------- EVALUATE ALL GUNS ----------
for g = 1:G
    if ~ishandle(wb), error('Progress bar closed.'); end

    f = str2func(names(g));
    Ettk = f(Q,H,D);  % expects Ettk as first output
    TTK(:,g) = reshape(Ettk, [N 1]);

    frac = g/G;
    elapsed = toc(t0);
    eta = elapsed*(1/frac - 1);

    if useGPU
        dev = gpuDevice;
        usedGB = (dev.TotalMemory - dev.AvailableMemory)/1e9;
        msg = sprintf('Gun %d/%d: %s\nElapsed %.1fs | ETA %.1fs\nGPU mem %.2f/%.2f GB', ...
            g, G, names(g), elapsed, eta, usedGB, dev.TotalMemory/1e9);
    else
        msg = sprintf('Gun %d/%d: %s\nElapsed %.1fs | ETA %.1fs', ...
            g, G, names(g), elapsed, eta);
    end

    waitbar(frac, wb, msg);
end
if ishandle(wb), delete(wb); end

%% ---------- TOP-3 TTK VALUES (NO SORT, GPU-FRIENDLY) ----------
[ttk1, idx1] = min(TTK, [], 2);

lin = (1:N).';
if useGPU, lin = gpuArray(lin); end

TTK2 = TTK;
TTK2(sub2ind([N,G], lin, idx1)) = inf;
[ttk2, idx2] = min(TTK2, [], 2);

TTK3 = TTK2;
TTK3(sub2ind([N,G], lin, idx2)) = inf;
[ttk3, idx3] = min(TTK3, [], 2);

% bring to CPU for plotting
ttk1 = gather(ttk1);
ttk2 = gather(ttk2);
ttk3 = gather(ttk3);

%% ---------- DIFFERENCES ----------
d12 = ttk2 - ttk1;  % rank2 - rank1
d23 = ttk3 - ttk2;  % rank3 - rank2

% If rank#2 or #3 is Inf, differences become Inf -> clamp to diffCap for black
d12(~isfinite(d12)) = diffCap;
d23(~isfinite(d23)) = diffCap;

% Clamp to [0, diffCap]
d12 = max(0, min(d12, diffCap));
d23 = max(0, min(d23, diffCap));

%% ---------- PLOTTING (same 3D axes, grayscale) ----------
Qp = Qcpu(:); Hp = Hcpu(:); Dp = Dcpu(:);

plotDiff3D(Qp, Hp, Dp, d12, diffCap, markerSize, ...
    sprintf('Spread: Rank#2 - Rank#1 (seconds) | white=0, black>=%.1fs | grid=%dx%dx%d', diffCap, sz(1),sz(2),sz(3)));

plotDiff3D(Qp, Hp, Dp, d23, diffCap, markerSize, ...
    sprintf('Spread: Rank#3 - Rank#2 (seconds) | white=0, black>=%.1fs | grid=%dx%dx%d', diffCap, sz(1),sz(2),sz(3)));

%% ---------- LOCAL FUNCTION ----------
function plotDiff3D(q,h,d,delta,cap,ms,titleTxt)
    figure('Color','w');
    scatter3(q, h, d, ms, delta, 'filled', ...
        'MarkerFaceAlpha', 0.9, 'MarkerEdgeAlpha', 0.9);

    grid on;
    xlabel('Accuracy q');
    ylabel('Headshot fraction h');
    zlabel('Distance (m)');
    title(titleTxt,'Interpreter','none');
    view(3);

    % Color mapping: 0 -> white, cap -> black
    colormap(flipud(gray(256)));
    caxis([0 cap]);
    cb = colorbar;
    ylabel(cb, 'Seconds');
end
