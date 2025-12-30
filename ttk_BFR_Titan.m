function [Ettk, pKill, Eb, Eshots] = ttk_BFR_Titan(q, h, dist_m)
%TTK_BFR_TITAN Expected TTK with reload cycling (uncapped, infinite mags), GPU-friendly.
%
% Gun:
%   HP=350, MAG=5, RPM=71.6
%   Body=90, Head=135
%   Falloff: <=30m => 1, 30-45 linear to 0.7, >=45 => 0.7
% Reload:
%   FULL_RELOAD = 4.75s (used when a mag ends and target still alive)
%
% Inputs can be scalars or arrays; works with gpuArray.
%
% Outputs:
%   Ettk   : expected time to kill (seconds), includes reload cycles
%   pKill  : P(kill within one mag)
%   Eb     : E[min(tau,MAG)] expected bullets within one mag (capped)
%   Eshots : expected bullets to kill with reload cycling (uncapped)

    % ---- gun constants ----
    HP  = 350;
    MAG = 5;
    RPM = 71.6;

    BODY = 90;
    HEAD = 135;

    % falloff
    MINR = 30;
    MAXR = 45;
    MINM = 0.7;

    % reload
    FULL_RELOAD = 4.75;   % seconds (edit here in-file)

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end

    % ---- broadcast to common size ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz,'like',q);
    h      = h      + zeros(sz,'like',h);
    dist_m = dist_m + zeros(sz,'like',dist_m);

    % ---- damage multiplier m(dist) ----
    m = ones(sz,'like',dist_m);
    m(dist_m >= MAXR) = MINM;
    mid = (dist_m > MINR) & (dist_m < MAXR);
    % linear drop 1 -> 0.7 over 15m => slope 0.02
    m(mid) = 1 - 0.02 .* (dist_m(mid) - MINR);

    % ---- unit-space trick ----
    % gcd(90,135)=45 => body=2 units, head=3 units where 1 unit=45*m damage
    BASE_GCD = 45;
    T = ceil(HP ./ (BASE_GCD .* m));

    % max threshold at min m
    Tmax = ceil(HP / (BASE_GCD * MINM)); % ceil(350/(45*0.7)) = 12
    T = min(max(T,1),Tmax);

    % ---- per-shot probabilities ----
    pM = 1 - q;
    pB = q .* (1 - h);
    pH = q .* h;

    % ---- vectorize over all elements ----
    M = numel(q);
    pM = reshape(pM,[M 1]);
    pB = reshape(pB,[M 1]);
    pH = reshape(pH,[M 1]);
    T  = reshape(T, [M 1]);

    % dp(:,u+1) = P(total_units == u), truncated to u < T
    dp = zeros(M, Tmax, 'like', q);
    dp(:,1) = 1;

    cols = cast(0:Tmax-1,'like',q);   % 0..Tmax-1
    aliveMask = cols < T;             % MxTmax logical
    dp = dp .* aliveMask;

    tail0  = ones(M,1,'like',q);      % alive after 0 shots
    Eb_vec = tail0;                   % sum tail[0..MAG-1]
    tail   = tail0;

    BODY_STEP = 2;
    HEAD_STEP = 3;

    for n = 1:MAG
        new = dp .* pM; % miss

        % body shift +2
        new(:,1+BODY_STEP:end) = new(:,1+BODY_STEP:end) + dp(:,1:end-BODY_STEP) .* pB;

        % head shift +3
        new(:,1+HEAD_STEP:end) = new(:,1+HEAD_STEP:end) + dp(:,1:end-HEAD_STEP) .* pH;

        new = new .* aliveMask;

        dp = new;
        tail = sum(dp,2);             % alive after n shots

        if n <= MAG-1
            Eb_vec = Eb_vec + tail;
        end
    end

    pKill_vec = 1 - tail;             % kill within one mag
    Eb_vec    = Eb_vec;               % E[min(tau,MAG)]

    % ---- convert to uncapped-with-reloads expectation ----
    dt = 60 / RPM;

    % Eshots = Eb + (1-pKill)*(MAG/pKill)   (Inf if pKill==0)
    Eshots_vec = inf(M,1,'like',q);
    ok = (pKill_vec > 0);
    Eshots_vec(ok) = Eb_vec(ok) + (1 - pKill_vec(ok)) .* (MAG ./ pKill_vec(ok));

    % Expected number of failed mags before success = (1-pKill)/pKill
    Efail_vec = zeros(M,1,'like',q);
    Efail_vec(ok) = (1 - pKill_vec(ok)) ./ pKill_vec(ok);
    Efail_vec(~ok) = inf;

    % TTK = dt*(Eshots-1) + reload_time * E[#failed mags]
    Ettk_vec = inf(M,1,'like',q);
    Ettk_vec(ok) = dt .* (Eshots_vec(ok) - 1) + FULL_RELOAD .* Efail_vec(ok);

    % ---- reshape back ----
    Ettk  = reshape(Ettk_vec,  sz);
    pKill = reshape(pKill_vec, sz);
    Eb    = reshape(Eb_vec,    sz);
    Eshots= reshape(Eshots_vec,sz);
end
