function [Ettk, pKill, EbMag, EbUncapped] = ttk_Lewis_Gun(q, h, dist_m)
%TTK_LEWIS_GUN Uncapped expected TTK with EMPTY reloads, GPU-friendly.
%
% Gun:
%   HP=350, MAG=47, RPM=500
%   Body=23, Head=34.5 (= 23*3/2 exactly)
%   Empty reload = 3.55s (ignore tactical reload)
% Falloff:
%   d<=35m => m=1
%   35<d<40 => linear to 0.67
%   d>=40m => m=0.67
%
% mm discretization:
%   mm = floor(dist_m*1000 + 1e-9)
%
% Outputs:
%   Ettk      : E[TTK] seconds (UNCAPPED, includes empty reloads)
%   pKill     : P(kill within 1 mag = 47 bullets)
%   EbMag     : E[min(tau,47)] capped expected bullets in one mag
%   EbUncapped: E[tau] uncapped expected bullets to kill

    % ---- constants ----
    MAG = 47;
    RPM = 500;

    EMPTY_RELOAD = 3.55;  % <-- edit this in-file any time

    % unit scaling:
    % body=23, head=34.5 => multiply by 2: body=46, head=69, HP=700
    % choose 1 unit = 23*m damage => body=+2 units, head=+3 units
    HP2 = 700;
    BASE = 23;
    BODY_STEP = 2;
    HEAD_STEP = 3;

    % Worst-case threshold at min m=0.67:
    % threshold = ceil(HP2 / (BASE*m)) = ceil(700 / (23*0.67)) = 46
    Tmax = 46;

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end

    % ---- broadcast to common size ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz, 'like', q);
    h      = h      + zeros(sz, 'like', h);
    dist_m = dist_m + zeros(sz, 'like', dist_m);

    % ---- multiplier as num/den using mm = floor(dist*1000+1e-9) ----
    % drop = 1 - 0.67 = 0.33 over 5m => slope = 0.066 per meter
    % for 35<d<40:
    %   m = 1 - 0.066*(d-35)
    % with mm:
    %   m = (661000 - 66*mm) / 1000000
    mm = floor(dist_m * 1000.0 + 1e-9);

    num = ones(sz, 'like', dist_m);
    den = ones(sz, 'like', dist_m);

    far = (mm >= 40000);
    mid = (mm > 35000) & (mm < 40000);

    % far: 0.67 = 67/100
    num(far) = 67;  den(far) = 100;

    % mid: (661000 - 66*mm)/1e6
    num(mid) = 661000 - 66 .* mm(mid);
    den(mid) = 1000000;

    % ---- threshold units ----
    % threshold = ceil(HP2*den / (BASE*num))
    T = ceil((HP2 .* den) ./ (BASE .* num));
    T = min(max(T, 1), Tmax);

    % ---- probabilities ----
    pM = 1 - q;
    pB = q .* (1 - h);
    pH = q .* h;

    % ---- vectorize over all elements ----
    M = numel(q);
    pM = reshape(pM, [M 1]);
    pB = reshape(pB, [M 1]);
    pH = reshape(pH, [M 1]);
    T  = reshape(T,  [M 1]);

    % dp(:,u+1) = P(total_units == u), only for u < T
    dp = zeros(M, Tmax, 'like', q);
    dp(:,1) = 1;

    cols = cast(0:Tmax-1, 'like', q);  % 0..45
    aliveMask = cols < T;              % MxTmax
    dp = dp .* aliveMask;

    % tail[n] = P(alive after n shots)
    tail0    = ones(M,1,'like',q);     % n=0
    EbMagVec = tail0;                   % sum_{n=0..MAG-1} tail[n]
    tail     = tail0;

    for n = 1:MAG
        new = dp .* pM;  % miss

        % body shift +2
        new(:,1+BODY_STEP:end) = new(:,1+BODY_STEP:end) + dp(:,1:end-BODY_STEP) .* pB;

        % head shift +3
        new(:,1+HEAD_STEP:end) = new(:,1+HEAD_STEP:end) + dp(:,1:end-HEAD_STEP) .* pH;

        new = new .* aliveMask; % truncate dead states

        dp = new;
        tail = sum(dp, 2);

        if n <= MAG-1
            EbMagVec = EbMagVec + tail;
        end
    end

    % pKill within one mag
    pKillVec = 1 - tail;

    % ---- uncapped expectations via repeated-mags identity ----
    % EbUncapped = EbMag / pKill
    % E[#empty reloads] = (1-pKill)/pKill
    EbUncappedVec = inf(M,1,'like',q);
    EReloadsVec   = inf(M,1,'like',q);

    ok = (pKillVec > 0);
    EbUncappedVec(ok) = EbMagVec(ok) ./ pKillVec(ok);
    EReloadsVec(ok)   = (1 - pKillVec(ok)) ./ pKillVec(ok);

    % ---- time ----
    dt = 60 / RPM;  % seconds between bullets, first bullet at t=0
    EttkVec = inf(M,1,'like',q);
    EttkVec(ok) = dt .* (EbUncappedVec(ok) - 1) + EMPTY_RELOAD .* EReloadsVec(ok);

    % reshape outputs
    Ettk      = reshape(EttkVec,        sz);
    pKill     = reshape(pKillVec,       sz);
    EbMag     = reshape(EbMagVec,       sz);
    EbUncapped= reshape(EbUncappedVec,  sz);
end
