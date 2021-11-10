%% EGITIM
clear;
clc;
giris=[0,0;0,1;1,0;1,1];
cikis=[0;1;1;0];
lr=0.5;
iterasyon=1;
katmanx=1;
katmany=1;
kontrol=true;
%-----------------------------
[~,cikis_boyutu]=size(cikis);%cikis_boyutu
[~,giris_boyutu]=size(giris); %giris_boyutu
[veri_sayisi,~]=size(giris); %toplam giris sayisi
%%------------------------------------------------


while  kontrol
    iterasyon=1;
    
    %katman_sayisi ayarlama----------------
    katman_sayisi=[giris_boyutu, katmanx, katmany, cikis_boyutu];
    [~,katman_index]=size(katman_sayisi);
    %---------------------------------------
    w={}; %weight
    b={}; %bias
    delta={};
    katmandaki_cikis={};
    katmandaki_giris={};
    
    
    %rastgele aðýrlýk atama-------
    for m=1:katman_index-1
        w{m}=rand(katman_sayisi(m+1),katman_sayisi(m));
        b{m}=rand(1,katman_sayisi(m+1));
    end
    
    
    while iterasyon<5000
        hata=0;
        
        %ileri_besleme------------------
        for k=1:veri_sayisi
            katmandaki_cikis{1} = giris(k,:);%örn 0,0
            
            for i=1:katman_index-1
                katmandaki_giris{i+1}=katmandaki_cikis{i}*w{i}'+b{i};
                katmandaki_cikis{i+1}=sigmoid(katmandaki_giris{i+1});
            end
            %--------------------------------
            output=katmandaki_cikis{end};
            hata=hata+sum((cikis(k,:)-output).^2); % 1 iterasyondaki hata
            delta{katman_index}=(cikis(k,:)-output).*output.*(1-output);
            
            for n=katman_index-1:-1:2
                delta{n}=delta{n+1}*w{n}.*katmandaki_cikis{n}.*(1-katmandaki_cikis{n});
            end
            
            for m=1:katman_index-1
                w{m}= w{m}+lr.*delta{m+1}'*katmandaki_cikis{m};
                b{m}= b{m} +lr .*delta{m+1};
            end
        end
        
        %disp(hata)
        iterasyon=iterasyon+1;
    end
    
    if hata > 0.001
        disp(hata)
        katmanx=katmanx+1;
        katmany=katmany+1;
        
        
    else
        kontrol=false;
        disp(hata)
        break;
    end
    
end


%% TEST
test=[1,0]
for i=1:katman_index-1
    test=test*w{i}'+b{i};
    test=sigmoid(test);
end

disp(test);



