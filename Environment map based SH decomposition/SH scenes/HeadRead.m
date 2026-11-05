%pp=hdrread('doge.hdr');
fileID = fopen('grace-new.hdr','r');
%C = textscan(fileID,'HeaderLines','%f %f %f %f','Delimiter',',');
InputText=textscan(fileID,'%s',20,'delimiter','\n')
fclose(fileID);
Intro=InputText{1};
disp(Intro)
