function [pStates,pSeq, fs, bs, s] = hmmdecode(seq,tr,e,varargin)


numStates = size(tr,1);
checkTr = size(tr,2);
if checkTr ~= numStates
    error(message('stats:CNNclassification:BadTransitions'));
end

% number of rows of e must be same as number of states

checkE  = size(e,1);
if checkE ~= numStates
    error(message('stats:CNNclassification:InputSizeMismatch'));
end

numSymbols = size(e,2);

% deal with options
if nargin > 3
    okargs = {'symbols'};
    symbols = internal.stats.parseArgs(okargs, {''}, varargin{:});
    
    if ~isempty(symbols)
        numSymbolNames = numel(symbols);
        if ~isvector(symbols) || numSymbolNames ~= numSymbols
            error(message('stats:CNNclassification:BadSymbols'));
        end
        [~, seq]  = ismember(seq,symbols);
        if any(seq(:)==0)
            error(message('stats:CNNclassification:MissingSymbol'));
        end
    end
end

if ~isnumeric(seq)
    error(message('stats:CNNclassification:MissingSymbolArg'));
end
numEmissions = size(e,2);
if any(seq(:)<1) || any(seq(:)~=round(seq(:))) || any(seq(:)>numEmissions)
     error(message('stats:CNNclassification:BadSequence', numEmissions));
end
seq = [numSymbols+1, seq ];
L = length(seq);

fs = zeros(numStates,L);
fs(1,1) = 1;  % assume that we start in state 1.
s = zeros(1,L);
s(1) = 1;
for count = 2:L
    for state = 1:numStates
        fs(state,count) = e(state,seq(count)) .* (sum(fs(:,count-1) .*tr(:,state)));
    end
    % scale factor normalizes sum(fs,count) to be 1. 
    s(count) =  sum(fs(:,count));
    fs(:,count) =  fs(:,count)./s(count);
end


bs = ones(numStates,L);
for count = L-1:-1:1
    for state = 1:numStates
      bs(state,count) = (1/s(count+1)) * sum( tr(state,:)'.* bs(:,count+1) .* e(:,seq(count+1))); 
    end
end


pSeq = sum(log(s));
pStates = fs.*bs;
 
pStates(:,1) = [];


