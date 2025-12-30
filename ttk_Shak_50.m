function [Ettk, pKill, EbMag, EbUncapped] = ttk_Shak_50(q, h, dist_m)
%TTK_SHAK50 Uncapped expected TTK with EMPTY reloads, GPU-friendly.
%
% Matches your Python damage model:
%   HP=350, MAG=20, RPM=420
%   Body total = 30, Head total = 45  (15x2 and 22.5x2)
%   Falloff:
%       d <= 15m : m = 1
%       15<d<25  : linear to 0.65
%       d >= 25m : m = 0.65
%   mm discretization: mm = floor(dist*1000 + 1e-9)
%
% Reload model:
%   EMPTY reload only (ignore tactical)
%
% Outputs:
%   Ettk      : E[TTK] seconds (UNCAPPED, includes empty reloads)
%   pKill     : P(kill within 1 mag = 20 firing events)
%   EbMag     : E[min(tau,20)] capped expected events in one mag
%   EbUncapped: E[tau] uncapped expected events to kill

    % ---- constants ----
    HP  = 350;
    MAG = 20;
    RPM = 420;

    EMPTY_RELOAD = 3.20;  % <-- edit this in-file any time

    % Unit-space trick (exact):
    % body=30, head=45 => gcd=15 => body_step=2, head_step=3
    BASE = 15;
    BODY_STEP = 2;
    HEAD_STEP = 3;

    % Worst-case multiplier is 0.65 => threshold ceil(350/(15*0.65)) = 36
    Tmax = 36;

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end

    % ---- broadcast to common size ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz, 'like', q);
    h      = h      + zeros(sz, 'like', h);
    dist_m = dist_m + zeros(sz, 'like', dist_m);

    % ---- multiplier fraction via mm = floor(dist*1000 + 1e-9) ----
    % Python equivalent:
    %   if mm <= 15000: num=1, den=1
    %   elif mm >= 25000: num=13, den=20
    %   else: num=305000 - 7*mm, den=200000
    mm = floor(dist_m * 1000.0 + 1e-9);

    num = ones(sz, 'like', dist_m);
    den = ones(sz, 'like', dist_m);

    far = (mm >= 25000);
    mid = (mm > 15000) & (mm < 25000);

    num(far) = 13;  den(far) = 20;
    num(mid) = 305000 - 7 .* mm(mid);
    den(mid) = 200000;

    % ---- threshold units ----
    % threshold = ceil(HP / (BASE*(num/den))) = ceil(HP*den/(BASE*num))
    T = ceil((HP .* den) ./ (BASE .* num));
    T = min(max(T, 1), Tmax);

    % ---- probabilities ----
    pM = 1 - q;
    pB = q .* (1 - h);
    pH = q .* h;

    % ---- reshape to [M x 1] ----
    M = numel(q);
    pM = reshape(pM, [M 1]);
    pB = reshape(pB, [M 1]);
    pH = reshape(pH, [M 1]);
    T  = reshape(T,  [M 1]);

    % dp(:,u+1) = P(total_units == u), truncated to u < T
    dp = zeros(M, Tmax, 'like', q);
    dp(:,1) = 1;

    cols = cast(0:Tmax-1, 'like', q);
    aliveMask = cols < T;      % MxTmax
    dp = dp .* aliveMask;

    tail0     = ones(M,1,'like',q);   % P(alive after 0)
    EbMag_vec = tail0;                % sum tail[0..MAG-1]
    tail      = tail0;

    for n = 1:MAG
        new = dp .* pM;  % miss

        % body shift +2
        new(:,1+BODY_STEP:end) = new(:,1+BODY_STEP:end) + dp(:,1:end-BODY_STEP) .* pB;

        % head shift +3
        new(:,1+HEAD_STEP:end) = new(:,1+HEAD_STEP:end) + dp(:,1:end-HEAD_STEP) .* pH;

        new = new .* aliveMask;  % truncate dead

        dp = new;
        tail = sum(dp, 2);       % alive after n

        if n <= MAG-1
            EbMag_vec = EbMag_vec + tail;
        end
    end

    % within-mag kill probability
    pKill_vec = 1 - tail;

    % ---- uncapped expectations via repeated-mags identity ----
    EbUncapped_vec = inf(M,1,'like',q);
    EReloads_vec   = inf(M,1,'like',q);

    ok = (pKill_vec > 0);
    EbUncapped_vec(ok) = EbMag_vec(ok) ./ pKill_vec(ok);           % E[tau]
    EReloads_vec(ok)   = (1 - pKill_vec(ok)) ./ pKill_vec(ok);     % expected empty reloads

    % ---- time ----
    dt = 60 / RPM;  % seconds between firing events, first at t=0
    Ettk_vec = inf(M,1,'like',q);
    Ettk_vec(ok) = dt .* (EbUncapped_vec(ok) - 1) + EMPTY_RELOAD .* EReloads_vec(ok);

    % ---- reshape outputs ----
    Ettk      = reshape(Ettk_vec,       sz);
    pKill     = reshape(pKill_vec,      sz);
    EbMag     = reshape(EbMag_vec,      sz);
    EbUncapped= reshape(EbUncapped_vec, sz);
end
