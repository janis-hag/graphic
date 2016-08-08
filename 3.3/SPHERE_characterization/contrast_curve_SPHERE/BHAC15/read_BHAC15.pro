function read_BHAC15,file, CUBE=CUBE, AGES=AGES
;+
;FUNCTION READ_BHAC15
;Purpose : read the evolutionary table of Barraffe t et al. 2015 named BHAC15_iso.2mass and place the values in a datacube
;
;INPUT : 
;file : the table filename
;
;OUTPUTS
;return a table with the first column corresponds to the age in Gyr and the remaining ones correspond to those stored into the original ascii table. 
;
;KEYWORDS
;CUBE: if true, then the return results is a 3 dimention table with X=the parameters (mass, log g, Teff...), the Y= the different masses, and Z are the ages.
;AGES: if the CUBE keyword is true, then a message gives the ages in Gyr associated to each plane (Z) of the cube.
;
;Version 1.0. Programmed by M. Bonnefoy on 26/03/2015.
;Version 2.0. M. Bonnefoy on 13/04/2015.
;-

struct=read_ascii(file,COMMENT_SYMBOL='!')

if stregex(file,'COND') ne -1 or stregex(file,'DUSTY') ne -1  then begin
t=[0.001,0.005,0.010,0.050,0.100,0.120,0.500,1.0,5.0,10.] ;ages
endif else begin
t=[0.0005,0.001,0.002,0.003,0.004,0.005,0.008,0.010,0.015,0.020,0.025,0.030,0.040,0.050,0.080,0.100,0.1200,0.200,0.300,0.400,0.500,0.625,0.800,1.0,2.0,3.0,4.0,5.0,8.0,10.] ;ages
endelse

nt=n_elements(t)

tabo=struct.field01
dim=size(tabo)
table=fltarr(dim(1)+1,dim(2))

rep=where(tabo(0,*)-shift(tabo(0,*),1) lt 0,nrep) ;where change in age


for i=0, nrep-2 do table[0,rep[i]:rep[i+1]-1]=t[i]
table[0,rep[nt-1]:dim(2)-1]=t[nt-1]
table[1:dim(1),*]=tabo

if keyword_set(CUBE) then begin
	tab=(table(1,*))[*]
	diffmass = tab(UNIQ(tab, SORT(tab)))
	nmass=n_elements(diffmass) ;number of different masses
	tb=replicate(!Values.F_NAN,dim(1),nmass,n_elements(t)) ;X=parameters,Y=masses,Z=ages
	for z=0, n_elements(t)-1 do begin
		for m=0, nmass-1 do begin
		mt=where(table(0,*) eq t[z] and table(1,*) eq diffmass[m],nline)
		if nline ne 0 then tb[*,m,z]=table(1:dim(1),mt)
		endfor
	endfor
	
	table=tb
	if keyword_set(AGES) ne 0 then begin
		message,'Ages reported into the 3rd plane of the cube (Gyr): ',/info
		print,t
	endif	
endif


return,table
end