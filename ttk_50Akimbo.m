function [Ettk, pKill, EbMag, EbUncapped] = ttk_50Akimbo(q, h, dist_m)
%TTK_50AKIMBO Uncapped expected TTK with EMPTY reloads, GPU-friendly.
%
% Gun: .50 Akimbo
%   HP=350, MAG=14, RPM=230
%   BODY=44, HEAD=88
%   Empty reload = 3.00s (we ignore tactical reload)
% Falloff (exact via mm):
%   d <= 32m : m = 1
%   32<d<39  : linear to 0.5
%   d >= 39m : m = 0.5
%
% Inputs can be scalars or arrays (supports gpuArray + implicit expansion).
%
% Outputs:
%   Ettk      : E[TTK] in seconds (UNCAPPED, includes empty reloads)
%   pKill     : P(kill within 1 mag = 14 bullets)
%   EbMag     : E[min(tau,14)]  (capped expected bullets within one mag)
%   EbUncapped: E[tau] (uncapped expected bullets to kill)

    % ---- constants ----
    HP  = 350;
    MAG = 14;
    RPM = 230;

    BODY = 44;
    HEAD = 88;

    EMPTY_RELOAD = 3.00;   % <-- edit this in-file any time

    MINR = 32;   % m
    MAXR = 39;   % m

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end

    % ---- broadcast to common size ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz, 'like', q);
    h      = h      + zeros(sz, 'like', h);
    dist_m = dist_m + zeros(sz, 'like', dist_m);

    % ---- exact falloff as fraction num/den using mm ----
    % mid region (32..39):
    %   m = 1 - 0.5*(d-32)/7
    % with mm:
    %   m = (46000 - mm)/14000
    mm = floor(dist_m * 1000.0 + 1e-9);

    num = ones(sz, 'like', dist_m);
    den = ones(sz, 'like', dist_m);

    far = (mm >= 39000);
    mid = (mm > 32000) & (mm < 39000);

    % far: 0.5 = 1/2
    num(far) = 1;  den(far) = 2;

    % mid: (46000 - mm)/14000
    num(mid) = 46000 - mm(mid);
    den(mid) = 14000;

    % ---- unit-space trick ----
    % Choose 1 unit = BODY * (num/den) damage
    % Then body_step=1, head_step=2 exactly (since HEAD=2*BODY).
    BODY_STEP = 1;
    HEAD_STEP = 2;

    % threshold units = ceil(HP*den / (BODY*num))
    T = ceil((HP .* den) ./ (BODY .* num));

    % Tmax occurs at min m = 0.5 -> ceil(350/(44*0.5)) = 16
    Tmax = 16;
    T = min(max(T, 1), Tmax);

    % ---- probabilities ----
    pM = 1 - q;
    pB = q .* (1 - h);
    pH = q .* h;

    % ---- vectorize ----
    M = numel(q);
    pM = reshape(pM, [M 1]);
    pB = reshape(pB, [M 1]);
    pH = reshape(pH, [M 1]);
    T  = reshape(T,  [M 1]);

    % dp(:,u+1) = P(total_units == u), truncated to u < T
    dp = zeros(M, Tmax, 'like', q);
    dp(:,1) = 1;

    cols = cast(0:Tmax-1, 'like', q);   % 0..15
    aliveMask = cols < T;               % MxTmax
    dp = dp .* aliveMask;

    % tail[n] = P(alive after n shots)
    tail0  = ones(M,1,'like',q);   % n=0
    EbMag_vec = tail0;             % sum_{n=0..MAG-1} tail[n]
    tail = tail0;

    for n = 1:MAG
        new = dp .* pM;  % miss

        % body shift +1
        new(:,2:end) = new(:,2:end) + dp(:,1:end-1) .* pB;

        % head shift +2
        new(:,3:end) = new(:,3:end) + dp(:,1:end-2) .* pH;

        new = new .* aliveMask;    % truncate dead states

        dp = new;
        tail = sum(dp, 2);

        if n <= MAG-1
            EbMag_vec = EbMag_vec + tail;
        end
    end

    % kill within mag
    pKill_vec = 1 - tail;                % P(kill by 14)

    % ---- uncapped expectations via "repeated mags" identity ----
    % If each mag is an independent attempt with success prob pKill:
    %   E[bullets to kill] = EbMag / pKill
    %   E[#empty reloads]  = (1 - pKill) / pKill
    EbUncapped_vec = inf(M,1,'like',q);
    Ereloads_vec   = inf(M,1,'like',q);

    ok = (pKill_vec > 0);
    EbUncapped_vec(ok) = EbMag_vec(ok) ./ pKill_vec(ok);
    Ereloads_vec(ok)   = (1 - pKill_vec(ok)) ./ pKill_vec(ok);

    % ---- time ----
    dt = 60 / RPM;  % seconds between bullets, first bullet at t=0
    Ettk_vec = inf(M,1,'like',q);
    Ettk_vec(ok) = dt .* (EbUncapped_vec(ok) - 1) + EMPTY_RELOAD .* Ereloads_vec(ok);

    % reshape outputs
    Ettk      = reshape(Ettk_vec,     sz);
    pKill     = reshape(pKill_vec,    sz);
    EbMag     = reshape(EbMag_vec,    sz);
    EbUncapped= reshape(EbUncapped_vec, sz);
end
