function [guessTR,guessE,logliks] = hmm(seqs,guessTR,guessE,varargin)
tol = 1e-6;
trtol = tol;
etol = tol;
maxiter = 500;
pseudoEcounts = false;
pseudoTRcounts = false;
verbose = false;
[numStates, checkTr] = size(guessTR);
if checkTr ~= numStates
    error(message('stats:hmm:BadTransitions'));
end

% number of rows of e must be same as number of states

[checkE, numEmissions] = size(guessE);
if checkE ~= numStates
    error(message('stats:hmm:InputSizeMismatch'));
end
if (numStates ==0 || numEmissions == 0)
    guessTR = [];
    guessE = [];
    return
end

baumwelch = true;

if nargin > 3
    if rem(nargin,2)== 0
        error(message('stats:hmm:WrongNumberArgs', mfilename));
    end
    okargs = {'symbols','tolerance','pseudoemissions','pseudotransitions','maxiterations','verbose','algorithm','trtol','etol'};
    dflts  = {[]        []         []                []                  maxiter         verbose   ''           []      []};
    [symbols,tol,pseudoE,pseudoTR,maxiter,verbose,alg,trtol,etol] = ...
        internal.stats.parseArgs(okargs, dflts, varargin{:});
    
    
    if ~isempty(symbols)
        numSymbolNames = numel(symbols);
        if ~isvector(symbols) || numSymbolNames ~= numEmissions
            error(message('stats:hmm:BadSymbols', numEmissions));
        end
        
        % deal with a single sequence first
        if ~iscell(seqs) || ischar(seqs{1})
            [~, seqs]  = ismember(seqs,symbols);
            if any(seqs(:)==0)
                error(message('stats:hmm:MissingSymbol'));
            end
        else  % now deal with a cell array of sequences
            numSeqs = numel(seqs);
            newSeqs = cell(numSeqs,1);
            for count = 1:numSeqs
                [~, newSeqs{count}] = ismember(seqs{count},symbols);
                if any(newSeqs{count}(:)==0)
                    error(message('stats:hmm:MissingSymbol'));
                end
            end
            seqs = newSeqs;
        end
    end
    if ~isempty(pseudoE)
        [rows, cols] = size(pseudoE);
        if  rows < numStates
            error(message('stats:hmm:BadPseudoEmissionsRows'));
        end
        if  cols < numEmissions
            error(message('stats:hmm:BadPseudoEmissionsCols'));
        end
        numStates = rows;
        numEmissions = cols;
        pseudoEcounts = true;
    end
    if ~isempty(pseudoTR)
        [rows, cols] = size(pseudoTR);
        if rows ~= cols
            error(message('stats:hmm:BadPseudoTransitions'));
        end
        if  rows < numStates
            error(message('stats:hmm:BadPseudoEmissionsSize'));
        end
        numStates = rows;
        pseudoTRcounts = true;
    end
    if ischar(verbose)
        verbose = any(strcmpi(verbose,{'on','true','yes'}));
    end
    
    if ~isempty(alg)
        alg = internal.stats.getParamVal(alg,{'baumwelch','viterbi'},'Algorithm');
        baumwelch = strcmpi(alg,'baumwelch');
    end
end

if isempty(tol)
    tol = 1e-6;
end
if isempty(trtol)
    trtol = tol;
end
if isempty(etol)
    etol = tol;
end


if isnumeric(seqs)
    [numSeqs, seqLength] = size(seqs);
    cellflag = false;
elseif iscell(seqs)
    numSeqs = numel(seqs);
    cellflag = true;
else
    error(message('stats:hmm:BadSequence'));
end

% initialize the counters
TR = zeros(size(guessTR));

if ~pseudoTRcounts
    pseudoTR = TR;
end
E = zeros(numStates,numEmissions);

if ~pseudoEcounts
    pseudoE = E;
end

converged = false;
loglik = 1; % loglik is the log likelihood of all sequences given the TR and E
logliks = zeros(1,maxiter);
for iteration = 1:maxiter
    oldLL = loglik;
    loglik = 0;
    oldGuessE = guessE;
    oldGuessTR = guessTR;
    for count = 1:numSeqs
        if cellflag
            seq = seqs{count};
            seqLength = length(seq);
        else
            seq = seqs(count,:);
        end
        
        if baumwelch   % Baum-Welch training
            % get the scaled forward and backward probabilities
            [~,logPseq,fs,bs,scale] = hmmdecode(seq,guessTR,guessE);
            loglik = loglik + logPseq;
            logf = log(fs);
            logb = log(bs);
            logGE = log(guessE);
            logGTR = log(guessTR);
            % f and b start at 0 so offset seq by one
            seq = [0 seq];
            
            for k = 1:numStates
                for l = 1:numStates
                    for i = 1:seqLength
                        TR(k,l) = TR(k,l) + exp( logf(k,i) + logGTR(k,l) + logGE(l,seq(i+1)) + logb(l,i+1))./scale(i+1);
                    end
                end
            end
            for k = 1:numStates
                for i = 1:numEmissions
                    pos = find(seq == i);
                    E(k,i) = E(k,i) + sum(exp(logf(k,pos)+logb(k,pos)));
                end
            end
        else  % Viterbi training
            [estimatedStates,logPseq]  = hmmviterbi(seq,guessTR,guessE);
            loglik = loglik + logPseq;
            % w = warning('off');
            [iterTR, iterE] = hmmestimate(seq,estimatedStates,'pseudoe',pseudoE,'pseudoTR',pseudoTR);
            %warning(w);
            % deal with any possible NaN values
            iterTR(isnan(iterTR)) = 0;
            iterE(isnan(iterE)) = 0;
            
            TR = TR + iterTR;
            E = E + iterE;
        end
    end
    totalEmissions = sum(E,2);
    totalTransitions = sum(TR,2);
    
    % avoid divide by zero warnings
    guessE = E./(repmat(totalEmissions,1,numEmissions));
    guessTR  = TR./(repmat(totalTransitions,1,numStates));
    % if any rows have zero transitions then assume that there are no
    % transitions out of the state.
    if any(totalTransitions == 0)
        noTransitionRows = find(totalTransitions == 0);
        guessTR(noTransitionRows,:) = 0;
        guessTR(sub2ind(size(guessTR),noTransitionRows,noTransitionRows)) = 1;
    end
    % clean up any remaining Nans
    guessTR(isnan(guessTR)) = 0;
    guessE(isnan(guessE)) = 0;
    
    if verbose
        if iteration == 1
            fprintf('%s\n',getString(message('stats:hmm:RelativeChanges')));
            fprintf('   Iteration       Log Lik    Transition     Emmission\n');
        else 
            fprintf('  %6d      %12g  %12g  %12g\n', iteration, ...
                (abs(loglik-oldLL)./(1+abs(oldLL))), ...
                norm(guessTR - oldGuessTR,inf)./numStates, ...
                norm(guessE - oldGuessE,inf)./numEmissions);
        end
    end
    % Durbin et al recommend loglik as the convergence criteria  -- we also
    % use change in TR and E. Use (undocumented) option trtol and
    % etol to set the convergence tolerance for these independently.
    %
    logliks(iteration) = loglik;
    if (abs(loglik-oldLL)/(1+abs(oldLL))) < tol
        if norm(guessTR - oldGuessTR,inf)/numStates < trtol
            if norm(guessE - oldGuessE,inf)/numEmissions < etol
                if verbose
                    fprintf('%s\n',getString(message('stats:hmm:ConvergedAfterIterations',iteration)))
                end
                converged = true;
                break
            end
        end
    end
    E =  pseudoE;
    TR = pseudoTR;
end
% if ~converged
%     warning(message('stats:hmm:NoConvergence', num2str( tol ), maxiter));
% end
logliks(logliks ==0) = [];
