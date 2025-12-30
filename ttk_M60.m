function [Ettk, pKill, EbMag, EbUncapped] = ttk_M60(q, h, dist_m)
%TTK_M60 Uncapped expected TTK with EMPTY reloads, GPU-friendly.
% Matches your Python exactly:
%   HP=350, MAG=70, RPM=580
%   Body=20, Head=30
%   Falloff: <=25m => 1, 25-35m linear to 0.5, >=35m => 0.5
%   mm discretization: mm = round(dist*1000)
% Unit-space:
%   1 unit = 10*m damage  => body=+2 units, head=+3 units
%   threshold = ceil(350 / (10*m)) = ceil(35*den/num)
%
% Reload model:
%   EMPTY reload only (ignore tactical; here they are same anyway)
%
% Outputs:
%   Ettk      : E[TTK] seconds (UNCAPPED, includes empty reloads)
%   pKill     : P(kill within 1 mag = 70 bullets)
%   EbMag     : E[min(tau,70)] expected capped bullets in one mag
%   EbUncapped: E[tau] expected bullets to kill (uncapped)

    % ---- constants ----
    HP  = 350; %#ok<NASGU>
    MAG = 70;
    RPM = 580;

    EMPTY_RELOAD = 3.55;  % <-- edit this in-file any time (empty reload)

    % unit-space steps (fixed)
    BODY_STEP = 2;
    HEAD_STEP = 3;

    % Worst case m=0.5 => threshold ceil(350/(10*0.5)) = 70
    Tmax = 70;

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end

    % ---- broadcast to common size ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz, 'like', q);
    h      = h      + zeros(sz, 'like', h);
    dist_m = dist_m + zeros(sz, 'like', dist_m);

    % ---- multiplier fraction (num/den), with mm = round(dist*1000) ----
    mm = round(dist_m * 1000.0);

    num = ones(sz, 'like', dist_m);
    den = ones(sz, 'like', dist_m);

    far = (mm >= 35000);
    mid = (mm > 25000) & (mm < 35000);

    num(far) = 1;  den(far) = 2;          % 0.5
    num(mid) = 35000 - mm(mid);           % (35000 - x)/20000
    den(mid) = 20000;

    % ---- threshold = ceil(35*den/num) ----
    T = ceil((35 .* den) ./ num);
    T = min(max(T, 1), Tmax);

    % ---- probabilities ----
    pM = 1 - q;
    pB = q .* (1 - h);
    pH = q .* h;

    % ---- reshape to [M x 1] for vectorized DP ----
    M  = numel(q);
    pM = reshape(pM, [M 1]);
    pB = reshape(pB, [M 1]);
    pH = reshape(pH, [M 1]);
    T  = reshape(T,  [M 1]);

    % dp(:,u+1) = P(total_units == u), truncated to u < T
    dp = zeros(M, Tmax, 'like', q);
    dp(:,1) = 1;

    cols = cast(0:Tmax-1, 'like', q);   % 0..69
    aliveMask = cols < T;               % MxTmax
    dp = dp .* aliveMask;

    tail0     = ones(M,1,'like',q);     % P(alive after 0)
    EbMag_vec = tail0;                  % sum tail[0..MAG-1]
    tail      = tail0;

    for n = 1:MAG
        new = dp .* pM;  % miss

        % body shift +2
        new(:,1+BODY_STEP:end) = new(:,1+BODY_STEP:end) + dp(:,1:end-BODY_STEP) .* pB;

        % head shift +3
        new(:,1+HEAD_STEP:end) = new(:,1+HEAD_STEP:end) + dp(:,1:end-HEAD_STEP) .* pH;

        new = new .* aliveMask;   % truncate dead

        dp = new;
        tail = sum(dp, 2);        % P(alive after n)

        if n <= MAG-1
            EbMag_vec = EbMag_vec + tail;
        end
    end

    % within-mag kill probability
    pKill_vec = 1 - tail;

    % ---- uncapped via repeated-mags identity ----
    EbUncapped_vec = inf(M,1,'like',q);
    EReloads_vec   = inf(M,1,'like',q);

    ok = (pKill_vec > 0);
    EbUncapped_vec(ok) = EbMag_vec(ok) ./ pKill_vec(ok);          % E[tau]
    EReloads_vec(ok)   = (1 - pKill_vec(ok)) ./ pKill_vec(ok);    % expected empty reloads

    % ---- time ----
    dt = 60 / RPM;   % seconds between shots, first at t=0
    Ettk_vec = inf(M,1,'like',q);
    Ettk_vec(ok) = dt .* (EbUncapped_vec(ok) - 1) + EMPTY_RELOAD .* EReloads_vec(ok);

    % ---- reshape outputs ----
    Ettk       = reshape(Ettk_vec,       sz);
    pKill      = reshape(pKill_vec,      sz);
    EbMag      = reshape(EbMag_vec,      sz);
    EbUncapped = reshape(EbUncapped_vec, sz);
end
