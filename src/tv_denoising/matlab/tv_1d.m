function [tv_sig, error] = tv_1d(sig, lambda, eps)
    addpath([cd '/TFOCS/']);
    
    A = eye(length(sig));
    b = sig;
    
    tv = linop_TV(size(sig), [], 'cvx');
    mu = lambda*norm(tv(sig),Inf);
    x0 = zeros(size(sig));

    opts = [];
    opts.restart = 1000;
    opts.maxIts = 2000;

    W = linop_TV(size(sig));
    normW = linop_TV(size(sig), [], 'norm');
    opts.normW2 = normW^2;
    z0 = [];   % we don't have a good guess for the dual
    [ tv_sig, error, optsOut ] = solver_sBPDN_W(A, W, b, eps, mu, x0(:), z0, opts);
end
