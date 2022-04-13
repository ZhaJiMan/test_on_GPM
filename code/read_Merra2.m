clc,clear all;

lat_min=30.;
lat_max=50.;
lon_min=100.;
lon_max=125.;

strPath_Merra2='/data04/0/Backup/zhuhx/Merra2_2014_2020_spring/2020/';
NC='MERRA2_400.tavg1_2d_aer_Nx.';
 
yearStart= 2020; yearEnd = 2020; 
monthStart= 3; monthEnd = 5;
dayStart=1; dayEnd = 31;

AOT_data_2020=nan*zeros(1,195); 

for i = yearStart:yearEnd
    for j = monthStart:monthEnd
       for k = dayStart : dayEnd           
            days = sprintf('%d%02d%02d',i,j,k );  
             files_Merra2 = dir(strcat( strPath_Merra2, NC, days,'*.nc4'));
             if isempty(files_Merra2)==1
                continue;
             end 
             
              for n=1: length(files_Merra2)  
                nc_n = strcat( strPath_Merra2, files_Merra2(n).name);  
                yyyyddmm_n=str2double(files_Merra2(n).name(28:35));
                
                lon = ncread(nc_n,'lon');
                lat = ncread(nc_n,'lat');
                TOTEXTTAU=ncread(nc_n,'TOTEXTTAU'); % Total Aerosol Extinction AOT [550 nm]
                OCEXTTAU = ncread(nc_n,'OCEXTTAU'); % Organic Carbon Extinction AOT [550 nm]
                DUEXTTAU = ncread(nc_n,'DUEXTTAU'); % Dust Extinction AOT [550 nm]
                DUEXTT25 = ncread(nc_n,'DUEXTT25'); % Dust Extinction AOT [550 nm] - PM 2.5
                BCEXTTAU = ncread(nc_n,'BCEXTTAU'); % Black Carbon Extinction AOT [550 nm]
                SUEXTTAU = ncread(nc_n,'SUEXTTAU'); % SO4 Extinction AOT [550 nm]
                SSEXTTAU = ncread(nc_n,'SSEXTTAU'); % Sea Salt Extinction AOT [550 nm]
                SSEXTT25 = ncread(nc_n,'SSEXTT25'); % Sea Salt Extinction AOT [550 nm] - PM 2.5
                
                lon_0=lon*ones(1,361);
                lat_0=ones(576,1)*lat';
                
                lat_1 = reshape(lat_0,[],1);
                lon_1 = reshape(lon_0,[],1);
                TOT_1= reshape(TOTEXTTAU,[],24);
                OC_1= reshape(OCEXTTAU,[],24);
                DUST_1= reshape(DUEXTTAU,[],24);
                DUST25_1= reshape(DUEXTT25,[],24);
                BC_1= reshape(BCEXTTAU,[],24);
                SO4_1= reshape(SUEXTTAU,[],24);
                SS_1= reshape(SSEXTTAU,[],24);
                SS25_1= reshape(SSEXTT25,[],24);                
                
 
               index=(lat_1>=lat_min & lat_1<=lat_max & lon_1>=lon_min & lon_1<=lon_max);
               lat_new=lat_1(index);
               lon_new=lon_1(index); 
               TOT_new=TOT_1(index,:); 
               OC_new=OC_1(index,:); 
               DUST_new=DUST_1(index,:); 
               DUST25_new=DUST25_1(index,:); 
               BC_new=BC_1(index,:); 
               SO4_new=SO4_1(index,:); 
               SS_new=SS_1(index,:); 
               SS25_new=SS25_1(index,:);              
               
               day_new=repmat(yyyyddmm_n,length(lon_new),1);
 
               AOT_n = [day_new,lon_new,lat_new,TOT_new, DUST_new, DUST25_new, OC_new,BC_new,SO4_new,SS_new,SS25_new];
               AOT_data_2020  = [AOT_data_2020;AOT_n];  
               
               end
                disp(files_Merra2(n).name); 
       end
   end
end

% day=1, lon=2, lat=3, TOT=4:27, DUST=28:51, PM=52:75, OC=76:99,
% BC=100:123, SO4=124:147, SS=148:171, SS25=172:195
AOT_data_2020(1,:)=[];        
 
 save('/d6/zhuhx/dust_precipitation/JING_JIN_JI/data/Spring/Merra2/AOT_data_2020.mat','AOT_data_2020','-v7.3');
 disp('Finshed');               

 
 
 
 
 
 
 
 
 
 
