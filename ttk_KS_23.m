function [Ettk, pKill, EbMag, EbUncapped] = ttk_KS_23(q, h, dist_m)
%TTK_KS_23 Uncapped expected TTK with EMPTY reloads, GPU-friendly.
%
% Gun (KS-23 slug model, body-only):
%   HP=350, MAG=6, RPM=73
%   Body=100 (head not modeled; h accepted but ignored)
%   Empty reload = 4.36s (ignore tactical reload)
%
% Falloff:
%   d<=18m => m=1
%   18<d<23 => linear to 0.7
%   d>=23m => m=0.7
%
% mm discretization:
%   mm = floor(dist*1000 + 1e-9)
%
% Outputs:
%   Ettk      : E[TTK] seconds (UNCAPPED, includes empty reloads)
%   pKill     : P(kill within 1 mag = 6 shots)
%   EbMag     : E[min(tau,6)] capped expected shots in one mag
%   EbUncapped: E[tau] uncapped expected shots to kill

    % ---- constants ----
    HP  = 350;
    MAG = 6;
    RPM = 73;

    BODY = 100;

    EMPTY_RELOAD = 4.36;  % <-- edit this in-file any time

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end % accepted but ignored

    % ---- broadcast to common size (include h to keep shapes consistent) ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz, 'like', q);
    h      = h      + zeros(sz, 'like', q); %#ok<NASGU>
    dist_m = dist_m + zeros(sz, 'like', dist_m);

    % ---- multiplier fraction via mm = floor(dist*1000 + 1e-9) ----
    mm = floor(dist_m * 1000.0 + 1e-9);

    num = ones(sz, 'like', dist_m);
    den = ones(sz, 'like', dist_m);

    far = (mm >= 23000);
    mid = (mm > 18000) & (mm < 23000);

    num(far) = 7;   den(far) = 10;     % 0.7

    % mid: m = (104000 - 3*mm)/50000
    num(mid) = 104000 - 3 .* mm(mid);
    den(mid) = 50000;

    % ---- threshold in "hit units" ----
    % 1 hit = BODY*(num/den) damage, threshold = ceil(HP*den / (BODY*num))
    T = ceil((HP .* den) ./ (BODY .* num));

    % worst case m=0.7 => ceil(350/(100*0.7)) = 5
    Tmax = 5;
    T = min(max(T, 1), Tmax);

    % ---- probabilities (body-only) ----
    pM   = 1 - q;
    pHit = q;

    % ---- reshape to [M x 1] ----
    M = numel(q);
    pM   = reshape(pM,   [M 1]);
    pHit = reshape(pHit, [M 1]);
    T    = reshape(T,    [M 1]);

    % dp(:,u+1) = P(total_hits == u), truncated to u < T
    dp = zeros(M, Tmax, 'like', q);
    dp(:,1) = 1;

    cols = cast(0:Tmax-1, 'like', q);  % 0..4
    aliveMask = cols < T;              % MxTmax
    dp = dp .* aliveMask;

    tail0    = ones(M,1,'like',q);     % P(alive after 0)
    EbMagVec = tail0;                  % sum_{n=0..MAG-1} tail[n]
    tail     = tail0;

    for n = 1:MAG
        new = dp .* pM;                % miss

        % hit shift +1
        new(:,2:end) = new(:,2:end) + dp(:,1:end-1) .* pHit;

        new = new .* aliveMask;        % truncate dead states

        dp = new;
        tail = sum(dp, 2);             % P(alive after n)

        if n <= MAG-1
            EbMagVec = EbMagVec + tail;
        end
    end

    % kill within one mag
    pKillVec = 1 - tail;

    % ---- uncapped expectations using repeated-mags identity ----
    EbUncappedVec = inf(M,1,'like',q);
    EReloadsVec   = inf(M,1,'like',q);

    ok = (pKillVec > 0);
    EbUncappedVec(ok) = EbMagVec(ok) ./ pKillVec(ok);          % E[tau]
    EReloadsVec(ok)   = (1 - pKillVec(ok)) ./ pKillVec(ok);    % expected empty reloads

    % ---- time ----
    dt = 60 / RPM;  % seconds between shots, first shot at t=0
    EttkVec = inf(M,1,'like',q);
    EttkVec(ok) = dt .* (EbUncappedVec(ok) - 1) + EMPTY_RELOAD .* EReloadsVec(ok);

    % ---- reshape outputs ----
    Ettk      = reshape(EttkVec,        sz);
    pKill     = reshape(pKillVec,       sz);
    EbMag     = reshape(EbMagVec,       sz);
    EbUncapped= reshape(EbUncappedVec,  sz);
end
