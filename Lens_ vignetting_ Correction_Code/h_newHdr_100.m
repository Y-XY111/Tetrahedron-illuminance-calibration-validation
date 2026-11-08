%clc ;
%clear;
% 创建新的hdr图像-下半部分
% vigetting correcting   等距投影矫正标准
Imagecorrecttingname="Z:\1028数据\1028数据\糖宝家 mixed 2500K&4500K\mixed 双鱼眼未拼接.hdr";
Imagecorrectting=hdrread(Imagecorrecttingname);
%[rows,columns,pages]=size(Imagecorrectting);

%% read the vignetting correction function
VF_down='screendownlens_100.xlsx';
[coefs_down]=xlsread(VF_down,'A1:F1');
coefs_size=length(coefs_down);
  
VF_up='backuplens_100.xlsx';
[coefs_up]=xlsread(VF_up,'A1:F1');


%cp=[round(columns/2), round(rows/2)];  中心点
%maxRad=sqrt(cp(1)^2+cp(2)^2);   最大半径

% VCimage=zeros (rows, columns, pages);  创建与原始图像大小相同的全0数组
% for i=1:rows 
%     for j=1:columns     当前像素点相对于中心点的偏移量
%         x=j-cp(1);
%         y=i-cp(2);
        
%% calculate the angle; apply the correction function;calculate illuminance
Calibrationfactor=1;
map_size=size(Imagecorrectting);
assert(mod(map_size(1),2)==0, '双鱼眼图像高度必须为偶数。');
assert(all(isfinite(coefs_up)) & all(isfinite(coefs_down)), '渐晕系数必须为有限数值。');

Cut_map_up=Imagecorrectting(1:round((map_size(1))/2),1:map_size(2),:);
Cut_map=Cut_map_up;
Cut_map_down=Imagecorrectting(round((map_size(1))/2)+1:map_size(1),1:map_size(2),:);
%Cut_map=Cut_map_down;
Cut_map_size=size(Cut_map);

center_pixel_x_up=Cut_map_size(2)/2;
center_pixel_y_up=Cut_map_size(1)/2;

center_pixel_x_down=Cut_map_size(2)/2;
center_pixel_y_down=Cut_map_size(1)+Cut_map_size(1)/2;

radiance=center_pixel_x_up;

fisheyeluminance=179*( 0.2126.*Imagecorrectting(:,:,1) + 0.7152.*Imagecorrectting(:,:,2) +0.0722.*Imagecorrectting(:,:,3))*Calibrationfactor;

% 保持原标定关系：0.9R 对应90度；仅将校正范围扩展至98度。
circle_r=center_pixel_x_up*0.9; % 照度积分半径（90度）
correction_angle_max=100; % 图像校正到100度
calibration_angle_max=98; % 98至100度沿用98度系数
f=7.2;  
single_pixel_size=f*pi/2/circle_r;
Ai=single_pixel_size^2;
%f_pixel focal lens in pixel
%radiance*0.9=f*pi/2
f_pixel=radiance*0.9/(pi/2);

outputImage =  Imagecorrectting; 
%outputImage = zeros(map_size(1), map_size(2), 3);     %新添加代码
  
  illuminance_pixel_up=zeros(Cut_map_size(1),Cut_map_size(2));
  illuminance_pixel_down=zeros(Cut_map_size(1),Cut_map_size(2));
%% 渐晕校正至100度；98至100度沿用98度系数；照度积分仅限90度内
% 100度之外保留原始RGB；98至100度使用各镜头自身的98度校正系数。
% 保留原来的圆心与等距投影参数，以匹配标定函数。
for lens_index=1:2
    if lens_index==1
        row_start=1;
        row_end=Cut_map_size(1);
        center_x=center_pixel_x_up;
        center_y=center_pixel_y_up;
        coefs=coefs_up;
    else
        row_start=Cut_map_size(1)+1;
        row_end=map_size(1);
        center_x=center_pixel_x_down;
        center_y=center_pixel_y_down;
        coefs=coefs_down;
    end

    for row=row_start:row_end
        for col=1:map_size(2)
            radius_pixel=sqrt((col-center_x)^2+(row-center_y)^2);
            theta=radius_pixel/f_pixel; % 弧度
            angle_deg=theta*180/pi;

            if angle_deg<=correction_angle_max
                angle_used=min(angle_deg,calibration_angle_max);
                V=polyval(coefs,angle_used);
                if ~isfinite(V) || V<=0
                    error('镜头%d，像素(%d,%d)，角度%.4f度的渐晕响应无效。', ...
                        lens_index,row,col,angle_deg);
                end
                CF=1/V;
                outputImage(row,col,:)=Imagecorrectting(row,col,:)*CF;

                % 90度外只校正图像，不参与平面照度积分。
                if radius_pixel<circle_r
                    if radius_pixel==0
                        d_w=Ai/f^2; % 中心处的极限，避免0/0
                    else
                        d=radius_pixel*single_pixel_size;
                        d_w=Ai*sin(theta)/(f*d);
                    end
                    contribution=fisheyeluminance(row,col)*d_w*cos(theta)*CF;
                    local_row=row-row_start+1;
                    if lens_index==1
                        illuminance_pixel_up(local_row,col)=contribution;
                    else
                        illuminance_pixel_down(local_row,col)=contribution;
                    end
                end
            end
        end
    end
end

illuminance_all_up=sum(illuminance_pixel_up(:))
illuminance_all_down=sum(illuminance_pixel_down(:))
[input_folder,input_stem]=fileparts(char(Imagecorrecttingname));
output_name=fullfile(input_folder,[input_stem '_vignetting_corrected.hdr']);
hdrwrite(outputImage,output_name);
fprintf('已保存100度渐晕校正图像（98至100度沿用98度系数）：%s\n',output_name);
