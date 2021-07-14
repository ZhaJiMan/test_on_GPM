;---------------------------------------------------------------------------
; Notes
; - 仅在NCL6.4.0下能无报错运行.
; - 将潜热的变量名改为latentHeating,缺测值改为-9999.9.
; - 潜热的高度随序数增大而增大.
; - 去掉了precipRate和airTemperature的保存,以减小文件体积.
; - 不把过高或过低的f3d设为缺测?
;---------------------------------------------------------------------------

;begin


; 参数改为从命令行中读取.
; bash中用单引号保护赋值语句.
; file1="/data00/0/GPM/DPR/V06/2018/201805/2A.GPM.DPR.V8-20180723.20180505-S043132-E060403.023764.V06A.HDF5"
; fienv="/data00/0/GPM/ENV/V06/2018/2A-ENV.GPM.DPR.V8-20180723.20180505-S043132-E060403.023764.V06A.HDF5"
; newfi="2A-LH.GPM.DPR.20180505-S043132-E060403.023764.nc";

fi := addfile(file1, "r")

;Constants
Lv      := 2.26e+6;      J/kg
Ls      := 2.59e+6;      J/kg
Cpa     := 1.01e+3;      J/kg/K
rhoa    := 1.29;         kg/m3
Va      := 1000;          m3
;aot     = (/-0.1567,0.6606,0/);
;afl     = (/0.5065,0.1456,0.0013/);
; aot     = (/-0.16,2.64,0/);
; afl     = (/-0.30,2.29,0/);

;Read data
fillv   = 9999.9;
setfileoption("h5", "FileStructure", "Advanced");
;DPR
gfi     = fi=>/NS;
slat    = gfi->Latitude;
slon    = gfi->Longitude;
rain3   = gfi->precipRate;
;bsur    = toint(gfi->binRealSurface);
elev    = gfi->elevation;


  peak = gfi->binBBPeak
  top = gfi->binBBTop
  bottom := gfi->binBBBottom
  stormtop:=gfi->binStormTop
;nbin=fi->nbin
elev    = elev/1000.0;      km

peak@_FillValue=-1111h
peak@_FillValue=0h

top@_FillValue=-1111h
top@_FillValue=0h

bottom@_FillValue=-1111h
bottom@_FillValue=0h

stormtop@_FillValue=-9999h

; printVarSummary(bottom)
; exit

dimrain=dimsizes(rain3)
;printVarSummary(peak)

rc=new((/dimrain(0),dimrain(1)/), double)
nbin=ispan(0,176,1)

 do k = 0, dimrain(0)-1
     do l = 0, dimrain(1)-1
        if(ismissing(peak(k,l))) then
            continue
       end if

       if(toint(stormtop(k,l)).gt.toint(top(k,l))-3) then
        continue
      end if

        p=tointeger(top(k,l)-3)
        ; print(doubletointeger(top(k,l)-2))
        ; exit
            q=tointeger(bottom(k,l)+1)
            ; print(num(.not. ismissing(rain3(k,l,p:q))))
            ;       exit
             if (num(.not. ismissing(rain3(k,l,p:q))).le.2) then
                continue
            end if
            ;print(p)
        do m = 0, dimrain(2)-1
            if (m.gt.bottom(k,l)+1.or.m.lt.top(k,l)-3) then
                continue
            end if

            if ((rain3(k,l,m)).eq.0.0) then
               rain3(k,l,m)=rain3@_FillValue
            end if

             if (ismissing(rain3(k,l,m))) then
                continue
            end if

            ; print(max(data(k,p:q)))
             ; if (rain3(k,l,m).eq.max(rain3(k,l,p:q))) then
             ;   rain3(k,l,m)=rain3@_FillValue
             ; end if

           rain3(k,l,m)=log10(rain3(k,l,m))
           end do

        ; write loop content

    ;   rc:=regline(data&height2(p:q),data(i,p:q))
   rc:=regline(tofloat(nbin(p:q)), rain3(k,l,p:q))
    do n= 0, 175
         if (n.gt.q.or.n.lt.p )then

              ;  print(k+","+l)
                continue
            end if
       rain3(k,l,n)=rc*nbin(n)+ rc@yintercept

      rain3(k,l,n)=10^(rain3(k,l,n))
   end do
 end do

    ; write loop content
end do
do k = 0, dimrain(0)-1
     do l = 0, dimrain(1)-1
        if(ismissing(peak(k,l))) then
            continue
       end if

       if(toint(stormtop(k,l)).ne.toint(top(k,l))-2) then
        continue
      end if

        p=tointeger(top(k,l)-2)
        ; print(doubletointeger(top(k,l)-2))
        ; exit
            q=tointeger(bottom(k,l)+1)
            ; print(num(.not. ismissing(rain3(k,l,p:q))))
            ;       exit
             if (num(.not. ismissing(rain3(k,l,p:q))).le.2) then
                continue
            end if
            ;print(p)
        do m = 0, dimrain(2)-1
            if (m.gt.bottom(k,l)+1.or.m.lt.top(k,l)-2) then
                continue
            end if

            if ((rain3(k,l,m)).eq.0.0) then
               rain3(k,l,m)=rain3@_FillValue
            end if

             if (ismissing(rain3(k,l,m))) then
                continue
            end if

            ; print(max(data(k,p:q)))
             ; if (rain3(k,l,m).eq.max(rain3(k,l,p:q))) then
             ;   rain3(k,l,m)=rain3@_FillValue
             ; end if

           rain3(k,l,m)=log10(rain3(k,l,m))
           end do

        ; write loop content

    ;   rc:=regline(data&height2(p:q),data(i,p:q))
   rc:=regline(tofloat(nbin(p:q)), rain3(k,l,p:q))
    do n= 0, 175
         if (n.gt.q.or.n.lt.p )then

              ;  print(k+","+l)
                continue
            end if
       rain3(k,l,n)=rc*nbin(n)+ rc@yintercept

      rain3(k,l,n)=10^(rain3(k,l,n))
   end do
 end do

    ; write loop content
end do


 do k = 0, dimrain(0)-1
     do l = 0, dimrain(1)-1
        if(ismissing(peak(k,l))) then
            continue
       end if
       if(toint(stormtop(k,l)).ne.toint(top(k,l)-1)) then
        continue
      end if

        p=tointeger(top(k,l)-1)

            q=tointeger(bottom(k,l)+1)
            ; print(num(.not. ismissing(rain3(k,l,p:q))))
            ;       exit
             if (num(.not. ismissing(rain3(k,l,p:q))).le.2) then
                continue
            end if
            ;print(p)
        do m = 0, dimrain(2)-1
            if (m.gt.bottom(k,l)+1.or.m.lt.top(k,l)-1) then
                continue
            end if
            if ((rain3(k,l,m)).eq.0.0) then
               rain3(k,l,m)=rain3@_FillValue
            end if
             if (ismissing(rain3(k,l,m))) then
                continue
            end if

            ; print(max(data(k,p:q)))
             ; if (rain3(k,l,m).eq.max(rain3(k,l,p:q))) then
             ;   rain3(k,l,m)=rain3@_FillValue
             ; end if

           rain3(k,l,m)=log10(rain3(k,l,m))
           end do

        ; write loop content

    ;   rc:=regline(data&height2(p:q),data(i,p:q))
   rc:=regline(tofloat(nbin(p:q)), rain3(k,l,p:q))
    do n= 0, 175
         if (n.gt.q.or.n.lt.p )then

              ;  print(k+","+l)
                continue
            end if
       rain3(k,l,n)=rc*nbin(n)+ rc@yintercept

      rain3(k,l,n)=10^(rain3(k,l,n))
    end do
 end do

    ; write loop content
end do

 do k = 0, dimrain(0)-1
     do l = 0, dimrain(1)-1
        if(ismissing(peak(k,l))) then
            continue
       end if
       if(toint(stormtop(k,l)).ne.toint(top(k,l))) then
        continue
      end if

        p=tointeger(top(k,l))

            q=tointeger(bottom(k,l)+1)
            ; print(num(.not. ismissing(rain3(k,l,p:q))))
            ;       exit
             if (num(.not. ismissing(rain3(k,l,p:q))).le.2) then
                continue
            end if
            ;print(p)
        do m = 0, dimrain(2)-1
            if (m.gt.bottom(k,l)+1.or.m.lt.top(k,l)) then
                continue
            end if
            if ((rain3(k,l,m)).eq.0.0) then
               rain3(k,l,m)=rain3@_FillValue
            end if
             if (ismissing(rain3(k,l,m))) then
                continue
            end if

            ; print(max(data(k,p:q)))
             ; if (rain3(k,l,m).eq.max(rain3(k,l,p:q))) then
             ;   rain3(k,l,m)=rain3@_FillValue
             ; end if

           rain3(k,l,m)=log10(rain3(k,l,m))
           end do

        ; write loop content

    ;   rc:=regline(data&height2(p:q),data(i,p:q))
   rc:=regline(tofloat(nbin(p:q)), rain3(k,l,p:q))
    do n= 0, 175
         if (n.gt.q.or.n.lt.p )then

              ;  print(k+","+l)
                continue
            end if
       rain3(k,l,n)=rc*nbin(n)+ rc@yintercept

      rain3(k,l,n)=10^(rain3(k,l,n))
    end do
 end do

    ; write loop content
end do


rain    = rain3(:,:,::-1);
rain@_FillValue     =-9999.9
dimr    = dimsizes(rain);
delete([/fi,gfi/]);


; exit
;if(isfilepresent(fienv))then

fi      = addfile(fienv,"r");
; else continue
; end if
; print(str_get_cols(filenamesenv(k),63 ,93))
; exit
gfi     = fi=>/NS;
stem2   = gfi->airTemperature;
stem    = stem2(:,:,::-1);
stem    = stem-273.15;          degC
delete(stem2);
stem@_FillValue     = fillv;
setfileoption("h5", "FileStructure", "Standard");
delete([/fienv,fi,gfi/]);

;Height
hint            := 0.125;    km
toph            := hint*(dimr(2)-1);    km
;print((/ toph /));
heir            = fspan(0.00,toph,dimr(2));
;hei2            = fspan(1.5,17.5,17);
; heis            = new(19,float);
; heis(0)         = 0.25 - 0.125;     km
; heis(1)         = 0.75 - 0.125;
; heis(2:)        = (/hei2/) - 0.125;

hdat        = new(dimr,float);          height
hgtd        = hdat;                     HGT
do i = 0,dimr(2)-1
    hdat(:,:,i)     = heir(i);
    hgtd(:,:,i)     = elev;
end do
;-dRr/dz
drr         = new(dimr,float);
i = 0;
drr(:,:,i)  = (rain(:,:,i+1) - rain(:,:,i))/hint;
i = dimr(2)-1;
drr(:,:,i)  = (rain(:,:,i) - rain(:,:,i-1))/hint;
do i = 1,dimr(2)-2
    ;print("layer " + i);
    drr(:,:,i)  = (rain(:,:,i+1) - rain(:,:,i-1))/(2*hint);
end do
drr@_FillValue      = fillv
drr     = -drr  ; -dRr/dz
drr         = where(hdat.ge.hgtd,drr,drr@_FillValue);
stem        = where(hdat.ge.hgtd,stem,stem@_FillValue);


;Parts of coef k
pck     = new(dimr,float,fillv);
x1      = pck;  x3 = 1-x1
x1      = 1.0+stem/38;
x1      = where(stem.gt.0,1,x1);
x1      = where(stem.lt.-38,0,x1);
pck     = (Lv*x1+Ls*(1-x1))/(Cpa*rhoa*Va);

;LH
awl     := (/-0.08,2.06/);warm layer, low
aws     := (/-0.13,2.28/);warm layer,slope
aml     := (/0.03,2.29/);mixing layer, low
ams     := (/0.07,2.27/);mixing layer,slope
amp     := (/0.08,2.42/);mixing layer,plateau
acl     := (/0.59,0.37/);cold layer, low
acs     := (/0.51,0.58/);cold layer,slope
acp     := (/0.61,0.36/);cold layer,plateau

a03d        = new(dimr,float,fillv);
a13d        = a03d;

a03d        = where((stem.ge.0.0).and.(hgtd.le.2.0),awl(0),a03d);
a13d        = where((stem.ge.0.0).and.(hgtd.le.2.0),awl(1),a13d);


a03d        = where((stem.ge.0.0).and.(hgtd.ge.2.0).and.(hgtd.le.4.0),aws(0),a03d);
a13d        = where((stem.ge.0.0).and.(hgtd.ge.2.0).and.(hgtd.le.4.0),aws(1),a13d);

a03d        = where((stem.ge.-38.0).and.(stem.le.0.0).and.(hgtd.le.2.0),aml(0),a03d);
a13d        = where((stem.ge.-38.0).and.(stem.le.0.0).and.(hgtd.le.2.0),aml(1),a13d);

a03d        = where((stem.ge.-38.0).and.(stem.le.0.0).and.(hgtd.ge.2.0).and.(hgtd.le.4.0),ams(0),a03d);
a13d        = where((stem.ge.-38.0).and.(stem.le.0.0).and.(hgtd.ge.2.0).and.(hgtd.le.4.0),ams(1),a13d);

a03d        = where((stem.ge.-38.0).and.(stem.le.0.0).and.(hgtd.ge.4.0),amp(0),a03d);
a13d        = where((stem.ge.-38.0).and.(stem.le.0.0).and.(hgtd.ge.4.0),amp(1),a13d);

a03d        = where((stem.le.-38.0).and.(hgtd.le.2.0),acl(0),a03d);
a13d        = where((stem.le.-38.0).and.(hgtd.le.2.0),acl(1),a13d);


a03d        = where((stem.le.-38.0).and.(hgtd.ge.2.0).and.(hgtd.le.4.0),acs(0),a03d);
a13d        = where((stem.le.-38.0).and.(hgtd.ge.2.0).and.(hgtd.le.4.0),acs(1),a13d);

a03d        = where((stem.le.-38.0).and.(hgtd.ge.4.0),acp(0),a03d);
a13d        = where((stem.le.-38.0).and.(hgtd.ge.4.0),acp(1),a13d);

delete([/hdat,hgtd/]);
f3d         = pck*drr;
;这里将过高或过低的f3d设为了缺测
f3d         = where(f3d.ge.-25.0 .and. f3d.le.37.5,f3d,f3d@_FillValue);
lhc         = where(rain.ge.0.1,a03d+a13d*f3d,0);
lhc         = where(ismissing(rain),lhc@_FillValue,lhc);
delete([/a03d,a13d,f3d/]);

;Attributes
copy_VarCoords(rain,lhc);
;printVarSummary(slh);
;printVarSummary(lhc);
;rain!2      = "heir";
;lhc!2       = "heir";
;slh!2       = "heis";
;csh!2       = "heis";
;heir!0      = "heir";
;heis!0      = "heis";

;Write in file
; fiout="/d4/huangch/LHretrieve_DPR/201408/2A-LH.GPM.DPR."+str_get_cols(filenamesdpr(j),66 ,96)
; newfi       = fiout + ".nc";
; if(isfilepresent(newfi))then
;     system("rm -f " + newfi);
; end if
;setfileoption("h5", "FileStructure", "Standard");
setfileoption("nc", "FileStructure", "Advanced");
setfileoption("nc","Format","NetCDF4");
f1              = addfile(newfi,"c");
setfileoption(f1,"DefineMode",True)

;Assign file attributes
fAtt               = True;
fAtt@title         = "Latent Heating from GPM products";
fAtt@source_file   = "2ADPR, 2ADPREVN";
fAtt@Conventions   = "None";
fAtt@creation_date = systemfunc ("date");
fileattdef(f1, fAtt);       copy file attributes

dimNames = (/rain!0, rain!1, rain!2/);
dimSizes = (/dimr(0), dimr(1), dimr(2)/);
dimUnlim = (/ False, False, False/);
filedimdef(f1, dimNames, dimSizes, dimUnlim);

delete([/rain@units,rain@CodeMissingValue,rain@DimensionNames/]);
rain@Description    = "Precipitation rate";
;filevardef(f1, "precipRate", typeof(rain), getvardims(rain))
;filevarattdef(f1,"precipRate", rain);   copy T attributes
;f1->precipRate  = (/ rain /);

delete([/stem@units,stem@CodeMissingValue,stem@DimensionNames/]);
stem@Description    = "Temperature";
stem@Units          = "degree Celcius";
;filevardef(f1, "T", typeof(stem), getvardims(stem))
;filevarattdef(f1,"T", stem);   copy T attributes
;f1->T           = (/ stem /);

; 将lhc的缺测值从9999.9改为-9999.9
lhc := where(ismissing(lhc), -9999.9, lhc)
copy_VarCoords(rain,lhc);
lhc@_FillValue = -9999.9

lhc@Units   = "K/hr";
lhc@Description     = "LH from the algorithm";
filevardef(f1, "latentHeating", typeof(lhc), getvardims(lhc))
filevarattdef(f1,"latentHeating", lhc);   copy T attributes
f1->latentHeating    = (/ lhc /);

delete([/slat@units,slat@CodeMissingValue,slat@DimensionNames/]);
f1->Latitude    = slat;
delete([/slon@units,slon@CodeMissingValue,slon@DimensionNames/]);
f1->Longitude   = slon;

heir!0      = "nbin";
heir@Units  = "km";
heir@Description    = "height for precipitation";
; delete(heis@_FillValue);
; heis!0      = "nlayer";
; heis@Units  = "km";
;heis@Description    = "height for SLH";
f1->height_r    = heir;
;f1->height_s    = heis;
;print(f1);

delete([/f1,slon,slat,rain,elev,stem,drr,pck,x1,lhc,newfi,dimr,heir/])

;end
