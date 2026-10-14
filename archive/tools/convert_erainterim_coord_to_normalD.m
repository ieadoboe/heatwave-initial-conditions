function[long3,long1,k,m,coordlong,coordlat]=convert_erainterim_coord_to_normal()

%filename='era5_daily_pressurelevels_may2015_sphum_1000hpa.nc' 
%filename='era5_Shum_May2015-test.nc'
filename='era5_daily_pressurelevels_may2017_sphum_1000hpa.nc'
%filename='era5_daily_pressurelevels_may2014_sphum_1000hpa.nc'

%filename='era5_0.125x0.125_May2015_specifichumidity_content_only_surface.nc'
%long_nr=15 % in total nx=15 (Avalon) 
%lat_nr=15 % in total ny=15 (Avalon)
% choose very small domain 30km *30km for St. John's airport
long_nr=16
lat_nr=16 

%long1 is the longitude varying from -180 to 180 or 180W-180E
%long3 is the longitude variable from 0 to 360 (all positive)


%To convert longitude from (0-360) to (-180 to 180)
%Matlab and fortran long1=mod((long3+180),360)-180
%Matlab long1=rem((long3+180),360)-180
%Fortran long1=modulo((long3+180),360)-180
% To convert longitude from (-180 to 180) to (0-360)
%Matlab lon3=mod(lon1,360)
%Fortran lon3=modulo(lon1,360)
%The �mod� function is Fortran is equivalent to �rem� function in Matlab.
%The �modulo� function in Fortran is equivalent to �mod� function in Matlab.


%filename='era_interim_0125X0125_november2015_daily_pressure_500hpa_TEMP.nc'


long3=ncread(filename,'longitude');
lat3=ncread(filename,'latitude');

%convertionformula
long1=rem((long3+180),360)-180;

conv=[long3,long1];

%k=find(long3==296.7500)
%m=find(lat3==60.2500)
%m=find(lat3==47.75);
m=find(lat3==49.75);

%k=find(long3==305.6250) % -54.3750 till -52.5000 (from west to ost)
%k=find(long3==305.0000) % era5 take -54.5000 to -52.50 (from west to east)

%k=find(long3==305.0000) % era5 take 
%k=find(long3==307.00)
k=find(long3==305.5000);
coordlong1=conv(k:k+long_nr,:);
coordlong=coordlong1(:,2);

%m=find(lat3==48.2500) % era5 48.2500 till 46.50 (from North to south)
coordlat=lat3(m:m+lat_nr,:);

%k=find(long3==73.7500)
%m=find(lat3==19.6250)
k
m

end