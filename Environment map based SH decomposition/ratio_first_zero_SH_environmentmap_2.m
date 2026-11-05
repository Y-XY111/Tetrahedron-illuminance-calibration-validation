clc;
clear;
%function spherical()

%Imagename={'SH scenes/uffizi-large.hdr','SH scenes/winter_lake_01_4k.hdr','SH scenes/hotel_room_4k.hdr','SH scenes/snowy_forest_path_02_4k.hdr'};
Imagename={"Z:\1028数据\使用场景\二楼阳台-N\二楼阳台理光拼接.hdr"};

for num=1:length(Imagename)
    
    
    Imageused=Imagename{num};
    Img=hdrread(Imageused);
    
    %luminance_map = 179 .* (0.265.*Img(:,:,1) + 0.670.*Img(:,:,2) + 0.065.*Img(:,:,3));
    luminance_map =179.*(0.2126.*Img(:,:,1) + 0.7152.*Img(:,:,2) +0.0722.*Img(:,:,3));    % the conversion equations for linear signals
    map_size=size(luminance_map);
    

    %% initialize         light source
    size1=map_size(1); %num of row
    size2=map_size(2); %num of coloum
    pix_source_R=size1/pi; 
         
    luminance_matrix=zeros(size1,size2,3);
   
    for i=1:size1
        theta_s=i/size1*pi;
        %n=sin(theta_s)*pix_source_R*2*pi;
        for j=1:size2
           % phi_s=(2*pi/size2)*(j-(size2+1)/2);
             phi_s=(2*pi/size2)*(size2-j);% %2pi---0
            %luminance=luminance_map(i,j)*(n/(2*pi*pix_source_R));
            luminance=luminance_map(i,j)*sin(theta_s);
            luminance_matrix(i,j,:)=[theta_s, phi_s, luminance];
        end
    end
        
    P=luminance_matrix(:,:,3);
    
    %% first 9 SH calculation
      
      ll=[0,1,1,1,2,2,2,2,2];
      mm=[0,-1,0,1,-2,-1,0,1,2]; 
    
      for i=1:9
          l=ll(i);
          m=mm(i);
          ylm(:,:,i)=y_l_m(l,m,luminance_matrix(:,:,1), luminance_matrix(:,:,2));
      end
         
    
      for j=1:9
        
          f_value(:,:,j)=ylm(:,:,j).*(P);
          f(num,j)=sum(sum(f_value(:,:,j))).*(1/pix_source_R^2);
        
         % f_value_noylm(:,:,j)=(illuminance_matrix(:,:,3)./max_pixel);
         % f_noylm(num,j)=sum(sum(f_value_noylm(:,:,j))).*(4*pi/pixel_num);
      end
      Scalar=(sqrt(pi)/2)*f(num, 1)
      fistorder_strength(num)=sqrt(f(num,2)^2+f(num,3)^2+f(num,4)^2);
      first_zero(num)=sqrt(f(num,2)^2+f(num,3)^2+f(num,4)^2)./f(num,1);% ratio between first order to zero order
     % Normalized_first_zero(num)=1-first_zero(num)/sqrt(3)
end
             