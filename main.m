%% main_ranked_3d_ttk_gpu_distinct_colors.m
% 3 plots: Rank #1 / #2 / #3 lowest expected TTK
% GPU compute + waitbar + strong, distinct colors

clear; clc;

%% ---------- SETTINGS ----------
useGPU   = true;
doPlots  = true;
markerSize = 9;

% Grid
q_vals    = linspace(0.10, 1.00, 46);
h_vals    = linspace(0.00, 1.00, 46);
dist_vals = linspace(0, 50, 51);

%% ---------- FIND GUN FUNCTIONS ----------
files = dir('ttk_*.m');
names = strings(0,1);
for k = 1:numel(files)
    [~, fn] = fileparts(files(k).name);
    if fn ~= "main" && ~contains(fn,"ranked")
        names(end+1,1) = string(fn);
    end
end
names = sort(names);
G = numel(names);
if G < 3
    error('Need at least 3 ttk_*.m gun functions.');
end

fprintf('Found %d guns:\n', G);
disp(names);

%% ---------- GRID ----------
[Qcpu, Hcpu, Dcpu] = ndgrid(q_vals, h_vals, dist_vals);
sz = size(Qcpu);
N  = numel(Qcpu);

% Validate once on CPU
assert(all(Qcpu(:)>=0 & Qcpu(:)<=1));
assert(all(Hcpu(:)>=0 & Hcpu(:)<=1));
assert(all(Dcpu(:)>=0));

%% ---------- GPU ----------
if useGPU
    dev = gpuDevice;
    fprintf('GPU: %s | %.2f GB\n', dev.Name, dev.TotalMemory/1e9);
    Q = gpuArray(single(Qcpu));
    H = gpuArray(single(Hcpu));
    D = gpuArray(single(Dcpu));
else
    Q = Qcpu; H = Hcpu; D = Dcpu;
end

%% ---------- ALLOCATE ----------
if useGPU
    TTK = gpuArray.nan(N,G,'single');
else
    TTK = nan(N,G,'single');
end

%% ---------- WAITBAR ----------
wb = waitbar(0,'Starting...','Name','Evaluating guns');
t0 = tic;

%% ---------- EVALUATE ----------
for g = 1:G
    f = str2func(names(g));
    Ettk = f(Q,H,D);
    TTK(:,g) = reshape(Ettk,[N 1]);

    frac = g/G;
    elapsed = toc(t0);
    eta = elapsed*(1/frac - 1);

    if useGPU
        dev = gpuDevice;
        memUsed = (dev.TotalMemory - dev.AvailableMemory)/1e9;
        msg = sprintf('%s\n%.1fs elapsed | %.1fs ETA\nGPU mem %.2f / %.2f GB', ...
            names(g), elapsed, eta, memUsed, dev.TotalMemory/1e9);
    else
        msg = sprintf('%s\n%.1fs elapsed | %.1fs ETA', names(g), elapsed, eta);
    end
    waitbar(frac, wb, msg);
end
delete(wb);

%% ---------- TOP-3 (NO SORT) ----------
[~, idx1] = min(TTK,[],2);
lin = (1:N).'; if useGPU, lin=gpuArray(lin); end

TTK2 = TTK;
TTK2(sub2ind([N,G],lin,idx1)) = inf;
[~, idx2] = min(TTK2,[],2);

TTK3 = TTK2;
TTK3(sub2ind([N,G],lin,idx2)) = inf;
[~, idx3] = min(TTK3,[],2);

idx1 = gather(idx1);
idx2 = gather(idx2);
idx3 = gather(idx3);

Qp = Qcpu(:); Hp = Hcpu(:); Dp = Dcpu(:);

%% ---------- DISTINCT COLOR PALETTE ----------
Cbase = [
    0.90 0.10 0.10  % red
    0.10 0.60 0.10  % green
    0.10 0.20 0.90  % blue
    0.85 0.50 0.00  % orange
    0.60 0.20 0.70  % purple
    0.00 0.70 0.70  % cyan
    0.90 0.00 0.60  % magenta
    0.55 0.35 0.05  % brown
    0.00 0.00 0.00  % black
    0.50 0.50 0.50  % gray
];

if G <= size(Cbase,1)
    C = Cbase(1:G,:);
else
    % fallback: evenly spaced HSV, fully saturated
    hsvC = hsv(G);
    hsvC(:,2) = 1; % saturation
    hsvC(:,3) = 0.95; % brightness
    C = hsv2rgb(hsvC);
end

%% ---------- PLOTS ----------
plotRank(Qp,Hp,Dp,idx1,names,C,markerSize,'Rank #1 – Lowest Expected TTK');
plotRank(Qp,Hp,Dp,idx2,names,C,markerSize,'Rank #2 – Second Lowest Expected TTK');
plotRank(Qp,Hp,Dp,idx3,names,C,markerSize,'Rank #3 – Third Lowest Expected TTK');

%% ---------- LOCAL ----------
function plotRank(q,h,d,idx,names,C,ms,titleTxt)
    figure('Color','w');
    hold on;
    G = numel(names);
    hLeg = gobjects(G,1);

    for g = 1:G
        m = idx==g;
        if any(m)
            hLeg(g) = scatter3(q(m),h(m),d(m),ms, ...
                'MarkerFaceColor',C(g,:), ...
                'MarkerEdgeColor',C(g,:), ...
                'MarkerFaceAlpha',0.9, ...
                'MarkerEdgeAlpha',0.9);
        else
            hLeg(g) = scatter3(nan,nan,nan,ms,'MarkerFaceColor',C(g,:));
        end
    end

    grid on;
    xlabel('Accuracy q');
    ylabel('Headshot fraction h');
    zlabel('Distance (m)');
    title(titleTxt,'Interpreter','none');
    view(3);
    legend(hLeg,cellstr(names),'Location','eastoutside','Interpreter','none');
end
