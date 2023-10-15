clc;
clear;
close all;
close all hidden
warning off;
[speech,path]=uigetfile('*','select an file');
speech=audioread([path,speech]);
plot(speech);title('input speech signal');
 
%%%%%%%%%%%%feature extraction%%%%%%%%%%%

eps=.00000000000001;
ufft = [1 2 3 4];
fprintf ('Loading data ...\n');
data_folder_contents = dir ('./TRAIN DATA1');
datafolder = cell(0,0);
person_index = 0;
max_coeffs = [-Inf -Inf -Inf];
min_coeffs = [ Inf  0  0];

for person=1:size(data_folder_contents,1)
    if (strcmp(data_folder_contents(person,1).name,'.') || ...
        strcmp(data_folder_contents(person,1).name,'..') || ...
        (data_folder_contents(person,1).isdir == 0))
        continue;
    end
    person_index = person_index+1;
    person_name = data_folder_contents(person,1).name;
    datafolder{1,person_index} = person_name;
    fprintf([person_name,' ']);
    person_folder_contents = dir(['./TRAIN DATA1/',person_name,'/*.wav']);    
    blk_cell = cell(0,0);
    for face_index=1:1
        I = audioread(['./TRAIN DATA1/',person_name,'/',person_folder_contents(ufft(face_index),1).name]);
        I = imresize(I,[104 100]);
        I = ordfilt2(I,1,true(3));        
        blk_index = 0;
        for blk_begin=1:100
            blk_index=blk_index+1;
            blk = I(blk_begin:blk_begin+4,:);            
            [U,S,V] = svd(double(blk));
            blk_coeffs = [U(1,1) S(1,1) S(2,2)];
            max_coeffs = max([max_coeffs;blk_coeffs]);
            min_coeffs = min([min_coeffs;blk_coeffs]);
            blk_cell{blk_index,face_index} = blk_coeffs;
        end
    end
    datafolder{2,person_index} = blk_cell;
    if (mod(person_index,10)==0)
        fprintf('\n');
    end
end
delta = (max_coeffs-min_coeffs)./([18 10 7]-eps);
minval = [min_coeffs;max_coeffs;delta];
for person_index=1:4
    for file_index=1:1
        for block_index=1:100
            blk_coeffs = datafolder{2,person_index}{block_index,file_index};
            min_coeffs = minval(1,:);
            delta_coeffs = minval(3,:);
            qt = floor((blk_coeffs-min_coeffs)./delta_coeffs);
            datafolder{3,person_index}{block_index,file_index} = qt;
            label = qt(1)*10*7+qt(2)*7+qt(3)+1;            
            datafolder{4,person_index}{block_index,file_index} = label;
        end
        datafolder{5,person_index}{1,file_index} = cell2mat(datafolder{4,person_index}(:,file_index));
    end
end

TRGUESS = ones(7,7) * eps;
TRGUESS(7,7) = 1;
for r=1:6
        TRGUESS(r,r) = 0.6;
        TRGUESS(r,r+1) = 0.4;    
end

EMITGUESS = (1/1260)*ones(7,1260);
 fprintf('\n');
fprintf ('Feature extraction ...\n');

for person_index=1:4
    fprintf([datafolder{1,person_index},' ']);
    seqmat = cell2mat(datafolder{5,person_index})';
    [ESTTR,ESTEMIT]=hmm(seqmat,TRGUESS,EMITGUESS,'Tolerance',.00001,'Maxiterations',100,'Algorithm', 'BaumWelch');
    ESTTR = max(ESTTR,eps);
    ESTEMIT = max(ESTEMIT,eps);
    datafolder{6,person_index}{1,1} = ESTTR;
    datafolder{6,person_index}{1,2} = ESTEMIT;
    if (mod(person_index,10)==0)
        fprintf('\n');
    end
end

% save DATABASE datafolder minval

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%classification%%%%%%%%%%%%%
I=speech;
I = imresize(I,[104 100]);
I = ordfilt2(I,1,true(3));
min_coeffs = minval(1,:);
max_coeffs = minval(2,:);
delta_coeffs = minval(3,:);
seq = zeros(1,10);
for blk_begin=1:10    
    blk = I(blk_begin:blk_begin+4,:);    
    [U,S,V] = svd(double(blk));
    blk_coeffs = [U(:,1) S(:,1) S(:,2)];
    blk_coeffs = max([blk_coeffs;min_coeffs]);        
    blk_coeffs = min([blk_coeffs;max_coeffs]);                    
    qt = floor((blk_coeffs-min_coeffs)./delta_coeffs);
    label = qt(1)*7*10+qt(2)*7+qt(3)+1;                   
    seq(1,blk_begin) = label;
end     

number_of_persons_in_database = size(datafolder,2);
results = zeros(1,number_of_persons_in_database);
for i=1:number_of_persons_in_database    
    Train = datafolder{6,i}{1,1};
    Tst = datafolder{6,i}{1,2};
    [ignore,logpseq] = hmmdecode(seq,Train,Tst);    
    P=exp(logpseq);
    results(1,i) = P;
end
[maxlogpseq,person_index] = max(results);
helpdlg(['The speech belongs to is ',datafolder{1,person_index}]);  


