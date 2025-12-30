function [Ettk, pKill, Eb] = ttk_M134(q, h, dist_m)
%TTK_M134 Expected capped TTK (cap=250 bullets), GPU-friendly (chunked).
%
% EXACT MODEL:
%   HP=350, MAG=250, RPM=1500
%   body=11, head=14.63 = 11*(133/100) exactly
%   falloff: <=30m => 1, 30-50 linear to 0.40, >=50 => 0.40
%   mm discretization: mm = floor(dist*1000 + 1e-9)  (matches your Python)
%   TTK = START_DELAY + (E[min(tau,MAG)] - 1) * dt
%
% This implementation is mathematically identical to the "big dp" version,
% but uses CHUNKING so it does not explode GPU memory.

    % ---- manual knobs (edit in THIS file) ----
    START_DELAY  = 0.7;   % seconds (spin-up delay)  <<< CHANGE THIS VALUE
    CHUNK_M      = 512;   % how many (q,h,d) points per chunk on GPU (256/512/1024)

    % ---- constants ----
    MAG = 250;
    RPM = 1500;

    % Unit choice (exact):
    % 1 unit = (11/100)*m(d) damage => body=100 units, head=133 units
    BODY_STEP = 100;
    HEAD_STEP = 133;

    % Worst-case m=0.4 => Tmax = ceil(35000*5/(11*2)) = 7955
    Tmax = 7955;

    % ---- checks (GPU-safe) ----
    if gather(any(q(:) < 0 | q(:) > 1)), error('q must be in [0,1]'); end
    if gather(any(h(:) < 0 | h(:) > 1)), error('h must be in [0,1]'); end
    if gather(any(dist_m(:) < 0)),       error('dist_m must be >= 0'); end
    if gather(START_DELAY < 0),          error('START_DELAY must be >= 0'); end

    % ---- broadcast to common size ----
    sz = size(q + h + dist_m);
    q      = q      + zeros(sz, 'like', q);
    h      = h      + zeros(sz, 'like', h);
    dist_m = dist_m + zeros(sz, 'like', dist_m);

    % ---- multiplier fraction m(d)=num/den (exact, from your Python) ----
    % m = 1                       if d<=30
    % m = 0.4=2/5                 if d>=50
    % m = (190000 - 3*mm)/100000  if 30<d<50, with mm=floor(d*1000+1e-9)
    mm = floor(dist_m * 1000.0 + 1e-9);

    num = ones(sz, 'like', dist_m);
    den = ones(sz, 'like', dist_m);

    far = (mm >= 50000);
    mid = (mm > 30000) & (mm < 50000);

    num(far) = 2;  den(far) = 5;
    num(mid) = 190000 - 3 .* mm(mid);
    den(mid) = 100000; % gcd reduction not needed: ratio is exact either way

    % ---- threshold in unit-space (exact) ----
    % threshold = ceil(35000*den / (11*num))
    T = ceil((35000 .* den) ./ (11 .* num));
    T = min(max(T, 1), Tmax);

    % ---- probabilities ----
    pM = 1 - q;
    pB = q .* (1 - h);
    pH = q .* h;

    % ---- flatten to vectors for chunking ----
    Mtot = numel(q);
    pM = reshape(pM, [Mtot 1]);
    pB = reshape(pB, [Mtot 1]);
    pH = reshape(pH, [Mtot 1]);
    T  = reshape(T,  [Mtot 1]);

    % ---- outputs (vector form, then reshape back) ----
    Ettk_v  = zeros(Mtot, 1, 'like', q);
    pKill_v = zeros(Mtot, 1, 'like', q);
    Eb_v    = zeros(Mtot, 1, 'like', q);

    dt = 60 / RPM;

    % ---- process in chunks to avoid MxTmax giant allocations ----
    for i0 = 1:CHUNK_M:Mtot
        i1 = min(Mtot, i0 + CHUNK_M - 1);
        msz = i1 - i0 + 1;

        pM_c = pM(i0:i1);
        pB_c = pB(i0:i1);
        pH_c = pH(i0:i1);
        T_c  = T(i0:i1);

        % dp(:,u+1) = P(total_units == u), truncated to u < T
        dp = zeros(msz, Tmax, 'like', q);
        dp(:,1) = 1;

        cols = cast(0:Tmax-1, 'like', q);
        aliveMask = cols < T_c;      % [msz x Tmax]
        dp = dp .* aliveMask;

        tail  = ones(msz,1,'like',q);   % P(alive after 0)
        Eb_c  = tail;                  % sum tail[0..MAG-1]

        for n = 1:MAG
            new = dp .* pM_c;  % miss

            % body shift +100
            new(:,1+BODY_STEP:end) = new(:,1+BODY_STEP:end) + dp(:,1:end-BODY_STEP) .* pB_c;

            % head shift +133
            new(:,1+HEAD_STEP:end) = new(:,1+HEAD_STEP:end) + dp(:,1:end-HEAD_STEP) .* pH_c;

            new = new .* aliveMask;     % truncate dead states

            dp = new;
            tail = sum(dp, 2);          % alive after n

            if n <= MAG-1
                Eb_c = Eb_c + tail;
            end
        end

        pKill_c = 1 - tail;
        Ettk_c  = START_DELAY + dt .* (Eb_c - 1);

        Ettk_v(i0:i1)  = Ettk_c;
        pKill_v(i0:i1) = pKill_c;
        Eb_v(i0:i1)    = Eb_c;
    end

    % ---- reshape back ----
    Ettk  = reshape(Ettk_v,  sz);
    pKill = reshape(pKill_v, sz);
    Eb    = reshape(Eb_v,    sz);
end
